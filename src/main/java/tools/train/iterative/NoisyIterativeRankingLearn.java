package tools.train.iterative;

import java.util.ArrayList;
import java.util.Arrays;
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
    
    // Stockage local des poids et des valeurs maximales pour la normalisation
    private double[] currentWeights;
    private double[] maxValues; 

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
        this.maxValues = null;
    }

    @Override
    public FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception {
        return new FunctionParameters(); // Non utilisé
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
        // Au départ, modèle linéaire simple (sera remplacé dès le 1er apprentissage)
        ISinglevariateFunction currentModel = new LinearScoreFunction(); 
        
        for (int t = 1; t <= maxIterations; t++) {
            
            // Notification pour le monitoring (ExperimentNoisyLETRID)
            NoisyLearnStep stepStart = new NoisyLearnStep(currentModel, preferenceHistory, t);
            getSupport().firePropertyChange("step", null, stepStart);

            double gamma = randomUtil.nextDouble();
            IAlternative[] pairToQuery = null;
            IAlternative preferred = null;
            boolean isExploration = (gamma < alpha);

            // --- 1. Phase de Sélection ---
            if (isExploration) {
                // Exploration : Safe-GUS cherche à réduire l'incertitude dans des zones sûres
                pairToQuery = safeGus.selectSafePair(currentModel, t);
                if (pairToQuery != null) {
                    preferred = noisyOracle.getNoisyPreferredAlternative(pairToQuery[0], pairToQuery[1], currentModel);
                }
            } else {
                // Exploitation : CorrectionStrategy cherche les erreurs passées
                pairToQuery = correctionStrategy.findMostInconsistentPair(preferenceHistory, currentModel);
                if (pairToQuery != null) {
                    preferred = noisyOracle.getNoisyPreferredAlternative(pairToQuery[0], pairToQuery[1], currentModel);
                }
            }
            
            // --- 2. Mise à jour de l'historique et Apprentissage ---
            if (pairToQuery != null) {
                // Mise à jour des bornes de normalisation avec les nouvelles données
                updateMaxValues(pairToQuery[0]);
                updateMaxValues(pairToQuery[1]);

                Ranking<IAlternative> newRanking;
                if (preferred != null) {
                    IAlternative other = preferred.equals(pairToQuery[0]) ? pairToQuery[1] : pairToQuery[0];
                    newRanking = new Ranking<>(new IAlternative[]{preferred, other}, new Double[]{1.0, 0.0});
                } else {
                    newRanking = new Ranking<>(new IAlternative[]{pairToQuery[0], pairToQuery[1]}, new Double[]{0.5, 0.5});
                }
                preferenceHistory.add(newRanking);
                
                // Apprentissage : Mise à jour des poids via Perceptron normalisé
                currentModel = doLearnStep(currentModel, preferenceHistory);
            }
            
            // Debug minimal
            // System.out.println("Iteration " + t + " completed.");
        }
        
        return currentModel;
    }
    
    /**
     * Met à jour les valeurs maximales observées pour chaque critère.
     * Indispensable pour que le Support (0-150) n'écrase pas la Confiance (0-1).
     */
    private void updateMaxValues(IAlternative alt) {
        double[] vec = alt.getVector();
        if (this.maxValues == null) {
            this.maxValues = new double[vec.length];
            Arrays.fill(this.maxValues, 1.0); // Évite division par 0 au départ
        }
        for (int i = 0; i < vec.length; i++) {
            if (Math.abs(vec[i]) > this.maxValues[i]) {
                this.maxValues[i] = Math.abs(vec[i]);
            }
        }
    }

    /**
     * Algorithme du Perceptron avec Normalisation des données et Learning Rate faible.
     */
    protected ISinglevariateFunction doLearnStep(ISinglevariateFunction current, List<Ranking<IAlternative>> history) {
        if (history.isEmpty()) return current;

        int dim = history.get(0).getObjects()[0].getVector().length;

        // 1. Initialisation ALÉATOIRE (Casse la symétrie et prouve l'apprentissage)
        if (this.currentWeights == null) {
            this.currentWeights = new double[dim];
            double sum = 0;
            for (int i = 0; i < dim; i++) {
                this.currentWeights[i] = randomUtil.nextDouble();
                sum += this.currentWeights[i];
            }
            // Normalisation initiale (somme = 1)
            for(int i=0; i<dim; i++) this.currentWeights[i] /= sum;
        }

        // 2. Paramètres d'apprentissage (MODIFIÉS pour stabilité)
        double learningRate = 0.001; // TRES FAIBLE pour éviter les oscillations (était 0.05)
        int epochs = 100;            // BEAUCOUP DE PASSES pour compenser le taux faible
        
        // On utilise le modèle normalisé pour faire les prédictions internes
        NormalizedLinearFunction tempModel = new NormalizedLinearFunction(currentWeights, maxValues);

        for (int e = 0; e < epochs; e++) {
            boolean converged = true;
            for (Ranking<IAlternative> rank : history) {
                IAlternative winner = rank.getObjects()[0];
                IAlternative loser = rank.getObjects()[1];

                Double[] scores = rank.getScores();
                if (scores != null && scores[0].equals(scores[1])) continue; // Ignore indifférence

                // Prédiction sur données normalisées
                double scoreW = tempModel.computeScore(winner);
                double scoreL = tempModel.computeScore(loser);

                // Si erreur (le modèle actuel préfère le perdant), on corrige
                if (scoreW <= scoreL) {
                    converged = false;
                    double[] vecW = winner.getVector();
                    double[] vecL = loser.getVector();

                    for (int i = 0; i < dim; i++) {
                        // Gradient normalisé par le max de chaque critère
                        // Cela permet à la Confiance d'avoir autant d'impact que le Support
                        double normW = vecW[i] / maxValues[i];
                        double normL = vecL[i] / maxValues[i];
                        
                        this.currentWeights[i] += learningRate * (normW - normL);
                        
                        // Contrainte : Poids positifs uniquement
                        if (this.currentWeights[i] < 0) this.currentWeights[i] = 0;
                    }
                    // Mise à jour du modèle temporaire pour la suite de la boucle
                    tempModel = new NormalizedLinearFunction(currentWeights, maxValues);
                }
            }
            if (converged) break;
        }

        // 3. Normalisation finale des poids (somme = 1)
        double sum = 0.0;
        for (double w : this.currentWeights) sum += w;
        if (sum > 0) {
            for (int i = 0; i < dim; i++) this.currentWeights[i] /= sum;
        } else {
             // Fallback si tous les poids tombent à 0 (rare)
             for (int i = 0; i < dim; i++) this.currentWeights[i] = 1.0 / dim;
        }

        // Retourne le modèle spécial qui sait gérer la normalisation
        return new NormalizedLinearFunction(this.currentWeights, this.maxValues);
    }

    /**
     * Classe interne : Modèle linéaire qui divise les entrées par maxValues avant de pondérer.
     * Score = Somme( w_i * (x_i / max_i) )
     */
    private static class NormalizedLinearFunction extends LinearScoreFunction {
        private final double[] maxValues;
        private final double[] weights;

        public NormalizedLinearFunction(double[] weights, double[] maxValues) {
            super(weights);
            this.weights = weights;
            this.maxValues = maxValues;
        }

        @Override
        public double computeScore(IAlternative alt) {
            double score = 0.0;
            double[] vec = alt.getVector();
            for (int i = 0; i < vec.length; i++) {
                double val = vec[i];
                // Protection contre division par zéro ou null
                double max = (maxValues != null && i < maxValues.length) ? maxValues[i] : 1.0;
                double normVal = (max != 0) ? val / max : 0;
                
                score += weights[i] * normVal;
            }
            return score;
        }
        
        // Compatibilité avec DecisionRule via le polymorphisme
        public double computeScore(DecisionRule rule) {
             return computeScore((IAlternative) rule);
        }
    }
}