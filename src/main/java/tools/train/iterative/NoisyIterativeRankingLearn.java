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
    
    // Stockage local des poids du modèle
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
        this.currentWeights = null; 
    }

    @Override
    public FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception {
        return new FunctionParameters();
    }

    @Override
    public FunctionParameters learn() throws Exception {
        ISinglevariateFunction finalFunction = runIterativeLearning();
        
        FunctionParameters params = new FunctionParameters();
        if (this.currentWeights != null) {
            params.setWeights(this.currentWeights);
            params.setNbCriteria(this.currentWeights.length);
            params.setFunctionType("linear");
        }
        return params;
    }
    
    private ISinglevariateFunction runIterativeLearning() {
        ISinglevariateFunction currentModel = new LinearScoreFunction(); 
        
        for (int t = 1; t <= maxIterations; t++) {
            
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
            
            // --- Mise à jour de l'historique et Apprentissage ---
            if (pairToQuery != null) {
                Ranking<IAlternative> newRanking;
                if (preferred != null) {
                    IAlternative other = preferred.equals(pairToQuery[0]) ? pairToQuery[1] : pairToQuery[0];
                    newRanking = new Ranking<>(new IAlternative[]{preferred, other}, new Double[]{1.0, 0.0});
                } else {
                    newRanking = new Ranking<>(new IAlternative[]{pairToQuery[0], pairToQuery[1]}, new Double[]{0.5, 0.5});
                }
                preferenceHistory.add(newRanking);
                
                // C'est ici que la magie opère : on met à jour les poids !
                currentModel = doLearnStep(currentModel, preferenceHistory);
            }
            
            System.out.println("Iteration " + t + " completed.");
        }
        
        return currentModel;
    }
    
    /**
     * Implémentation du Perceptron pour apprendre les poids.
     */
    protected ISinglevariateFunction doLearnStep(ISinglevariateFunction current, List<Ranking<IAlternative>> history) {
        if (history.isEmpty()) return current;

        // 1. Initialisation des poids à la première itération
        if (this.currentWeights == null) {
            IAlternative ref = history.get(0).getObjects()[0];
            int dim = ref.getVector().length;
            this.currentWeights = new double[dim];
            for (int i = 0; i < dim; i++) {
                this.currentWeights[i] = 1.0 / dim; // Poids uniformes au début
            }
        }

        // 2. Algorithme d'apprentissage (Perceptron)
        double learningRate = 0.05;
        int epochs = 20; // On repasse 20 fois sur l'historique pour bien apprendre
        int dim = this.currentWeights.length;

        // On crée un modèle temporaire pour tester les prédictions
        LinearScoreFunction tempModel = new LinearScoreFunction(this.currentWeights);

        for (int e = 0; e < epochs; e++) {
            boolean converged = true;
            for (Ranking<IAlternative> rank : history) {
                IAlternative winner = rank.getObjects()[0];
                IAlternative loser = rank.getObjects()[1];

                Double[] scores = rank.getScores();
                if (scores != null && scores[0].equals(scores[1])) continue; // On ignore les cas d'indifférence

                double scoreW = tempModel.computeScore(winner);
                double scoreL = tempModel.computeScore(loser);

                // Si le modèle se trompe (donne un meilleur score au perdant)
                if (scoreW <= scoreL) {
                    converged = false;
                    double[] vecW = winner.getVector();
                    double[] vecL = loser.getVector();

                    // Mise à jour des poids : on les pousse vers le gagnant
                    for (int i = 0; i < dim; i++) {
                        this.currentWeights[i] += learningRate * (vecW[i] - vecL[i]);
                        // On garde les poids positifs (optionnel mais souvent mieux pour l'interprétation)
                        if (this.currentWeights[i] < 0) this.currentWeights[i] = 0;
                    }
                    // Mise à jour du modèle temporaire avec les nouveaux poids
                    tempModel = new LinearScoreFunction(this.currentWeights);
                }
            }
            if (converged) break; // Si tout est bon, on arrête
        }

        // 3. Normalisation (pour que la somme des poids fasse 1)
        double sum = 0.0;
        for (double w : this.currentWeights) sum += w;
        if (sum > 0) {
            for (int i = 0; i < dim; i++) this.currentWeights[i] /= sum;
        } else {
             for (int i = 0; i < dim; i++) this.currentWeights[i] = 1.0 / dim;
        }

        // 4. On retourne le nouveau modèle mis à jour
        return new LinearScoreFunction(this.currentWeights);
    }
}