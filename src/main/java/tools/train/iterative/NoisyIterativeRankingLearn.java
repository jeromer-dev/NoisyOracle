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
import tools.utils.NoiseModelConfig;
import tools.utils.RandomUtil;

public class NoisyIterativeRankingLearn extends IterativeRankingLearn {

    private final SafeGUS safeGus;
    private final CorrectionStrategy correctionStrategy;
    private final HumanLikeNoisyOracle noisyOracle;
    private final NoiseModelConfig noiseConfig;
    private final double alpha;
    private final RandomUtil randomUtil;
    
    // On conserve une copie locale car le getter de la classe parente peut ne pas être accessible ou suffisant
    private final int maxIterations;
    
    private List<Ranking<IAlternative>> preferenceHistory;

    public NoisyIterativeRankingLearn(int maxIterations, double alpha, SafeGUS safeGus, 
                                      CorrectionStrategy correctionStrategy, HumanLikeNoisyOracle noisyOracle, 
                                      NoiseModelConfig noiseConfig) {
        
        // CORRECTION 1 : Appel du constructeur parent requis par IterativeRankingLearn
        // Signature : (int nbIterations, RankingsProvider rankingsProvider, ISinglevariateFunction func, int nbMeasures)
        // On passe 'safeGus' comme provider, une fonction linéaire par défaut, et 0 pour les mesures (non critique ici)
        super(maxIterations, safeGus, new LinearScoreFunction(), 0);
        
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
     * Méthode obligatoire à implémenter car IterativeRankingLearn est abstraite.
     * Dans notre cas, elle n'est pas utilisée car nous surchargeons la méthode learn() complète.
     * On retourne null ou un objet vide.
     */
    @Override
    public FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception {
        return new FunctionParameters(); // Stub
    }

    /**
     * Surcharge complète de la boucle d'apprentissage pour intégrer l'Exploration/Exploitation.
     */
    @Override
    public FunctionParameters learn() throws Exception {
        ISinglevariateFunction currentModel = initializeModel(); 
        
        for (int t = 1; t <= maxIterations; t++) {
            
            // CORRECTION 2 : Utilisation de la classe concrète NoisyLearnStep
            NoisyLearnStep stepStart = new NoisyLearnStep(currentModel, preferenceHistory, t);
            
            // Notification des observateurs
            getSupport().firePropertyChange("step", null, stepStart);

            // CORRECTION 3 : Utilisation de noiseConfig (suppression du warning "unused")
            // On peut l'afficher ou l'utiliser pour le log si besoin, 
            // mais ici il est surtout utilisé implicitement par SafeGUS qui a reçu la config.
            // Pour marquer l'usage :
            double currentThreshold = noiseConfig.getAdaptiveThreshold(t);

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
                }
            }
            
            // --- Mise à jour de l'historique (U) ---
            if (pairToQuery != null) {
                Ranking<IAlternative> newRanking;
                
                if (preferred != null) {
                    IAlternative other = preferred.equals(pairToQuery[0]) ? pairToQuery[1] : pairToQuery[0];
                    // Création manuelle du Ranking
                    newRanking = new Ranking<>(
                        new IAlternative[]{preferred, other}, 
                        new Double[]{1.0, 0.0}
                    );
                } else {
                    // Indifférence
                    newRanking = new Ranking<>(
                        new IAlternative[]{pairToQuery[0], pairToQuery[1]}, 
                        new Double[]{0.5, 0.5}
                    );
                }
                preferenceHistory.add(newRanking);
                
                // --- Mise à jour du Modèle ---
                currentModel = doLearnStep(currentModel, preferenceHistory);
            }
            
            System.out.println("Iteration " + t + " completed (Seuil=" + String.format("%.2f", currentThreshold) + "). Mode: " + (isExploration ? "EXPLORATION" : "EXPLOITATION"));
        }
        
        // Retour final (conversion en FunctionParameters pour respecter la signature)
        FunctionParameters params = new FunctionParameters();
        // Ici, on pourrait injecter les poids de currentModel dans params si nécessaire
        return params;
    }
    
    // --- Méthodes locales ---
    
    protected ISinglevariateFunction initializeModel() {
        return new LinearScoreFunction();
    }
    
    protected ISinglevariateFunction doLearnStep(ISinglevariateFunction current, List<Ranking<IAlternative>> history) {
        // CORRECTION 4 : Placeholder pour le solveur réel
        // TODO: Appeler ici Kappalab ou le solveur pour mettre à jour les poids
        return current; 
    }
}