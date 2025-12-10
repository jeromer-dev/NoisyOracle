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

/**
 * Implémentation de l'algorithme principal Noisy-LETRID.
 * Étend IterativeRankingLearn pour hériter de la logique d'apprentissage de base.
 */
public class NoisyIterativeRankingLearn extends IterativeRankingLearn {

    private final SafeGUS safeGus;
    private final CorrectionStrategy correctionStrategy;
    private final HumanLikeNoisyOracle noisyOracle;
    private final NoiseModelConfig noiseConfig;
    private final double alpha;
    private final RandomUtil randomUtil;
    
    // On gère maxIterations en local car le parent IterativeRankingLearn ne l'expose pas forcément ainsi
    private final int maxIterations;
    
    private List<Ranking<IAlternative>> preferenceHistory;

    public NoisyIterativeRankingLearn(int maxIterations, double alpha, SafeGUS safeGus, 
                                      CorrectionStrategy correctionStrategy, HumanLikeNoisyOracle noisyOracle, 
                                      NoiseModelConfig noiseConfig) {
        super(); // Appel au constructeur par défaut de IterativeRankingLearn
        this.maxIterations = maxIterations;
        this.alpha = alpha;
        this.safeGus = safeGus;
        this.correctionStrategy = correctionStrategy;
        this.noisyOracle = noisyOracle;
        this.noiseConfig = noiseConfig;
        this.randomUtil = new RandomUtil();
        this.preferenceHistory = new ArrayList<>();
    }

    /**
     * Implémentation de la méthode learn() requise par AbstractRankingLearning.
     */
    @Override
    public FunctionParameters learn() throws Exception {
        ISinglevariateFunction finalModel = runIterativeLearning();
        
        // Conversion du résultat : on retourne un FunctionParameters vide ou adapté
        // car la signature de la classe mère l'exige.
        FunctionParameters params = new FunctionParameters();
        // Optionnel : remplir params avec les valeurs de finalModel si possible
        return params;
    }
    
    /**
     * Logique principale de Noisy-LETRID.
     */
    private ISinglevariateFunction runIterativeLearning() {
        
        ISinglevariateFunction currentModel = initializeModel(); 
        
        for (int t = 1; t <= maxIterations; t++) {
            
            // Notification de progression (utile pour l'ExperimentNoisyLETRID)
            LearnStep stepStart = new LearnStep(currentModel, preferenceHistory, t);
            getSupport().firePropertyChange("step", null, stepStart);

            double gamma = randomUtil.nextDouble();
            IAlternative[] pairToQuery = null;
            IAlternative preferred = null;
            boolean isExploration = (gamma < alpha);

            // --- Phase de Sélection ---
            if (isExploration) {
                // Exploration : Safe-GUS
                pairToQuery = safeGus.selectSafePair(currentModel, t);
                
                if (pairToQuery != null) {
                    preferred = noisyOracle.getNoisyPreferredAlternative(pairToQuery[0], pairToQuery[1], currentModel);
                } else {
                    System.out.println("Exploration skipped at t=" + t + ": No safe pair found.");
                }
            } else {
                // Exploitation : Correction Strategy
                pairToQuery = correctionStrategy.findMostInconsistentPair(preferenceHistory, currentModel);

                if (pairToQuery != null) {
                    preferred = noisyOracle.getNoisyPreferredAlternative(pairToQuery[0], pairToQuery[1], currentModel);
                } else {
                    // Si pas de paire incohérente, on peut fallback sur de l'exploration
                    // System.out.println("Exploitation skipped at t=" + t + ": No inconsistent pair found.");
                }
            }
            
            // --- Mise à jour de l'historique (U) ---
            if (pairToQuery != null) {
                Ranking<IAlternative> newRanking;
                
                if (preferred != null) {
                    // Création MANUELLE du Ranking (Index 0 = Préféré, Index 1 = Autre)
                    // Cette méthode marche quelle que soit l'implémentation de Ranking
                    IAlternative other = preferred.equals(pairToQuery[0]) ? pairToQuery[1] : pairToQuery[0];
                    newRanking = new Ranking<>(
                        new IAlternative[]{preferred, other}, 
                        new Double[]{1.0, 0.0}
                    );
                } else {
                    // Création MANUELLE d'indifférence
                    newRanking = new Ranking<>(
                        new IAlternative[]{pairToQuery[0], pairToQuery[1]}, 
                        new Double[]{0.5, 0.5}
                    );
                }
                preferenceHistory.add(newRanking);
                
                // --- Mise à jour du Modèle ---
                // On utilise la méthode locale qui simule ou appelle le vrai solveur
                currentModel = doLearnStep(currentModel, preferenceHistory);
            }
            
            System.out.println("Iteration " + t + " completed. Mode: " + (isExploration ? "EXPLORATION" : "EXPLOITATION"));
        }
        
        return currentModel;
    }
    
    // --- Méthodes locales ---
    
    protected ISinglevariateFunction initializeModel() {
        return new LinearScoreFunction();
    }
    
    /**
     * Simule l'étape d'apprentissage.
     * C'est ici que vous connecterez plus tard Kappalab ou votre solveur.
     */
    protected ISinglevariateFunction doLearnStep(ISinglevariateFunction current, List<Ranking<IAlternative>> history) {
        // TODO: Appeler ici le solveur réel pour mettre à jour les poids en fonction de 'history'.
        // Pour l'instant, on retourne le modèle courant pour permettre la compilation et l'exécution de la boucle.
        return current; 
    }
}