package tools.train.iterative;

import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.oracles.HumanLikeNoisyOracle; 
import tools.ranking.Ranking;
import tools.ranking.heuristics.CorrectionStrategy;
import tools.ranking.heuristics.SafeGUS;
import tools.train.AbstractRankingLearning;
import tools.train.LearnStep;
import tools.utils.NoiseModelConfig;
import tools.utils.RandomUtil;
import java.util.List;

/**
 * Implémentation de l'algorithme principal Noisy-LETRID (Algorithme 1 du projet).
 * Gère le compromis entre Exploration (Safe-GUS) et Exploitation (CorrectionStrategy) via le paramètre alpha.
 * Il étend AbstractRankingLearning pour utiliser le mécanisme de mise à jour du modèle d'apprentissage.
 */
public class NoisyIterativeRankingLearn extends AbstractRankingLearning {

    private final SafeGUS safeGus;
    private final CorrectionStrategy correctionStrategy;
    private final HumanLikeNoisyOracle noisyOracle;
    private final NoiseModelConfig noiseConfig;
    private final double alpha; // Paramètre de compromis alpha (Exploration vs Exploitation)
    private final RandomUtil randomUtil;
    
    // Pour stocker l'historique des préférences (U dans l'Algorithme 1)
    private List<Ranking<IAlternative>> preferenceHistory;


    public NoisyIterativeRankingLearn(int maxIterations, double alpha, SafeGUS safeGus, 
                                      CorrectionStrategy correctionStrategy, HumanLikeNoisyOracle noisyOracle, 
                                      NoiseModelConfig noiseConfig) {
        
        // Utilise le constructeur de la classe parente
        super(maxIterations); 
        this.alpha = alpha;
        this.safeGus = safeGus;
        this.correctionStrategy = correctionStrategy;
        this.noisyOracle = noisyOracle;
        this.noiseConfig = noiseConfig;
        this.randomUtil = new RandomUtil();
        this.preferenceHistory = new java.util.ArrayList<>();
    }


    /**
     * Exécute la boucle principale de l'apprentissage actif.
     * Implémente la logique de l'Algorithme 1 (Noisy-LETRID).
     */
    @Override
    public ISinglevariateFunction learn() {
        
        // Étape 2 (Initialisation)
        ISinglevariateFunction currentModel = initializeModel(); 
        
        // Boucle 4: while t < L do
        for (int t = 1; t <= getMaxIterations(); t++) {
            
            // 5: Calcul du seuil de sécurité adaptatif tau_t
            double adaptiveThreshold = noiseConfig.getAdaptiveThreshold(t);

            // 6: Tirer gamma ~ U[0,1]
            double gamma = randomUtil.nextDouble();

            IAlternative[] pairToQuery = null;
            IAlternative preferred = null;
            boolean isExploration = (gamma < alpha);

            if (isExploration) {
                // 7-11: Phase Exploration (Safe-GUS)
                
                // 8: (Ri, Rj) <- Noisy-GUS(gt-1, tau_t)
                pairToQuery = safeGus.selectSafePair(currentModel, t);
                
                if (pairToQuery != null) {
                    // 9: Ry <- Oracle(Ri, Rj)
                    preferred = noisyOracle.getNoisyPreferredAlternative(pairToQuery[0], pairToQuery[1], currentModel);
                } else {
                    // Si SafeGUS ne trouve rien, on passe
                    System.out.println("Exploration skipped at t=" + t + ": No safe pair found.");
                    continue;
                }
                
            } else {
                // 12-16: Phase Exploitation (Correction)

                // 13: (Ri, Rj) <- CorrectionStrategy(U, gt-1)
                pairToQuery = correctionStrategy.findMostInconsistentPair(preferenceHistory, currentModel);

                if (pairToQuery != null) {
                    // 14: y_new <- Oracle(Ri, Rj) (Re-vérification)
                    preferred = noisyOracle.getNoisyPreferredAlternative(pairToQuery[0], pairToQuery[1], currentModel);
                    // L'entrée est ajoutée pour mise à jour du modèle
                } else {
                    // Si CorrectionStrategy ne trouve rien, on passe
                    System.out.println("Exploitation skipped at t=" + t + ": No inconsistent pair found.");
                    continue;
                }
            }
            
            // Enregistrement de la requête / correction dans l'historique U
            if (preferred != null) {
                // 10: U <- U U {Ry}
                preferenceHistory.add(Ranking.fromPreferred(pairToQuery[0], pairToQuery[1], preferred));
            } else if (pairToQuery != null) {
                // Si l'oracle est indifférent (preferred == null)
                preferenceHistory.add(Ranking.fromIndifferent(pairToQuery[0], pairToQuery[1]));
            }
            
            // 17: gt <- ApprendreModèle(U)
            LearnStep step = new LearnStep(currentModel, preferenceHistory, t);
            currentModel = this.doLearnStep(step); // Utilise la méthode de la classe parente
            
            System.out.println("Iteration " + t + " completed. Mode: " + (isExploration ? "EXPLORATION" : "EXPLOITATION") + ". U size: " + preferenceHistory.size());
        }
        
        // 20: return gL
        return currentModel;
    }
    
    // Nous conservons les méthodes abstraites pour la compatibilité avec AbstractRankingLearning
    
    @Override
    protected ISinglevariateFunction initializeModel() {
        // La classe parente AbstractRankingLearning implémente probablement ceci.
        // Si elle est abstraite, vous devrez la fournir dans la classe parente.
        return super.initializeModel(); 
    }
    
    @Override
    protected ISinglevariateFunction doLearnStep(LearnStep step) {
        // La classe parente AbstractRankingLearning implémente probablement ceci.
        return super.doLearnStep(step);
    }
}