package tools.train.iterative;

import java.util.ArrayList;
import java.util.List;

import tools.alternatives.IAlternative;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.oracles.HumanLikeNoisyOracle;
import tools.ranking.Ranking;
import tools.ranking.heuristics.CorrectionStrategy;
import tools.ranking.heuristics.SafeGUS;
import tools.rules.DecisionRule;
import tools.train.IterativeRankingLearn;
import tools.train.LearnStep;
import tools.utils.NoiseModelConfig;
import tools.utils.RandomUtil;

public class NoisyIterativeRankingLearn extends IterativeRankingLearn {

    private final SafeGUS safeGus;
    private final CorrectionStrategy correctionStrategy;
    private final HumanLikeNoisyOracle noisyOracle;
    private final NoiseModelConfig noiseConfig;
    private final double alpha;
    private final RandomUtil randomUtil;
    
    private final int maxIterations;
    
    private List<Ranking<IAlternative>> preferenceHistory;
    
    // Stockage local des poids car LinearScoreFunction ne permet pas de les modifier/lire facilement
    private double[] currentWeights;

    public NoisyIterativeRankingLearn(int maxIterations, double alpha, SafeGUS safeGus, 
                                      CorrectionStrategy correctionStrategy, HumanLikeNoisyOracle noisyOracle, 
                                      NoiseModelConfig noiseConfig) {
        super(maxIterations, safeGus, new LinearScoreFunction(), 0);
        this.maxIterations = maxIterations;
        this.alpha = alpha;
        this.safeGus = safeGus;
        this.correctionStrategy = correctionStrategy;
        this.noisyOracle = noisyOracle;
        this.noiseConfig = noiseConfig;
        this.randomUtil = new RandomUtil();
        this.preferenceHistory = new ArrayList<>();
        this.currentWeights = null; // Sera initialisé à la première itération
    }

    @Override
    public FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception {
        // Non utilisé dans cette implémentation car on surcharge learn()
        return new FunctionParameters();
    }

    @Override
    public FunctionParameters learn() throws Exception {
        ISinglevariateFunction finalFunction = runIterativeLearning();
        
        // Construction de l'objet de retour FunctionParameters
        FunctionParameters params = new FunctionParameters();
        if (this.currentWeights != null) {
            params.setWeights(this.currentWeights);
            params.setNbCriteria(this.currentWeights.length);
            params.setFunctionType("linear");
        }
        return params;
    }
    
    private ISinglevariateFunction runIterativeLearning() {
        // Modèle initial (sans poids, ou poids uniformes si initialisé)
        ISinglevariateFunction currentModel = new LinearScoreFunction(); 
        
        for (int t = 1; t <= maxIterations; t++) {
            
            // Notification
            NoisyLearnStep stepStart = new NoisyLearnStep(currentModel, preferenceHistory, t);
            getSupport().firePropertyChange("step", null, stepStart);

            double gamma = randomUtil.nextDouble();
            IAlternative[] pairToQuery = null;
            IAlternative preferred = null;
            boolean isExploration = (gamma < alpha);

            // --- Phase de Sélection ---
            if (isExploration) {
                pairToQuery = safeGus.selectSafePair(currentModel, t);
                if (pairToQuery != null) {
                    preferred = noisyOracle.getNoisyPreferredAlternative(pairToQuery[0], pairToQuery[1], currentModel);
                }
            } else {
                pairToQuery = correctionStrategy.findMostInconsistentPair(preferenceHistory, currentModel);
                if (pairToQuery != null) {
                    preferred = noisyOracle.getNoisyPreferredAlternative(pairToQuery[0], pairToQuery[1], currentModel);
                }
            }
            
            // --- Mise à jour de l'historique et du modèle ---
            if (pairToQuery != null) {
                Ranking<IAlternative> newRanking;
                if (preferred != null) {
                    IAlternative other = preferred.equals(pairToQuery[0]) ? pairToQuery[1] : pairToQuery[0];
                    newRanking = new Ranking<>(new IAlternative[]{preferred, other}, new Double[]{1.0, 0.0});
                } else {
                    newRanking = new Ranking<>(new IAlternative[]{pairToQuery[0], pairToQuery[1]}, new Double[]{0.5, 0.5});
                }
                preferenceHistory.add(newRanking);
                
                // Apprentissage : met à jour currentWeights et recrée le modèle
                currentModel = doLearnStep(currentModel, preferenceHistory);
            }
            
            // Log simple (commenté pour éviter le spam, décommenter si besoin)
            // System.out.println("Iteration " + t + " [" + (isExploration ? "EXPL" : "EXPLO") + "]");
        }
        
        return currentModel;
    }
    
    /**
     * Algorithme du Perceptron pour mettre à jour les poids linéaires.
     */
    protected ISinglevariateFunction doLearnStep(ISinglevariateFunction current, List<Ranking<IAlternative>> history) {
        if (history.isEmpty()) return current;

        // 1. Initialisation des poids si nécessaire (au premier appel)
        if (this.currentWeights == null) {
            // On récupère la dimension depuis la première règle de l'historique
            IAlternative ref = history.get(0).getObjects()[0];
            int dim = ref.getVector().length;
            this.currentWeights = new double[dim];
            // Initialisation uniforme
            for (int i = 0; i < dim; i++) {
                this.currentWeights[i] = 1.0 / dim;
            }
        }

        // 2. Apprentissage (Perceptron / Gradient Descent simple)
        double learningRate = 0.05;
        int epochs = 20;
        int dim = this.currentWeights.length;

        // On recrée une fonction temporaire pour évaluer les scores avec les poids courants
        LinearScoreFunction tempModel = new LinearScoreFunction(this.currentWeights);

        for (int e = 0; e < epochs; e++) {
            boolean converged = true;
            for (Ranking<IAlternative> rank : history) {
                IAlternative winner = rank.getObjects()[0];
                IAlternative loser = rank.getObjects()[1];

                Double[] scores = rank.getScores();
                if (scores != null && scores[0].equals(scores[1])) continue; // Indifférence

                double scoreW = tempModel.computeScore(winner);
                double scoreL = tempModel.computeScore(loser);

                // Si erreur (le gagnant n'est pas devant)
                if (scoreW <= scoreL) {
                    converged = false;
                    double[] vecW = winner.getVector();
                    double[] vecL = loser.getVector();

                    for (int i = 0; i < dim; i++) {
                        // W_new = W_old + rate * (Winner - Loser)
                        this.currentWeights[i] += learningRate * (vecW[i] - vecL[i]);
                        // Contrainte de positivité
                        if (this.currentWeights[i] < 0) this.currentWeights[i] = 0;
                    }
                    // Mise à jour du modèle temporaire pour la prochaine comparaison
                    tempModel = new LinearScoreFunction(this.currentWeights);
                }
            }
            if (converged) break;
        }

        // 3. Normalisation (Somme = 1)
        double sum = 0.0;
        for (double w : this.currentWeights) sum += w;
        if (sum > 0) {
            for (int i = 0; i < dim; i++) this.currentWeights[i] /= sum;
        } else {
             // Si tout est à 0, reset uniforme
             for (int i = 0; i < dim; i++) this.currentWeights[i] = 1.0 / dim;
        }

        // 4. Retourner une nouvelle fonction avec les poids mis à jour
        return new LinearScoreFunction(this.currentWeights);
    }
}