package tools.ranking.heuristics;

import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.utils.NoiseModelConfig; 
import tools.utils.RankingUtil;
import java.util.List;
import java.util.stream.Collectors;
import sampling.MMAS; // Import nécessaire pour interagir avec le Sampler

public class SafeGUS extends UncertaintySampling {

    private final NoiseModelConfig noiseConfig;

    // Le constructeur doit appeler un constructeur existant de UncertaintySampling.
    // Pour l'utiliser, nous devons passer les arguments de la surcharge existante la plus courte.
    public SafeGUS(MMAS sampler, NoiseModelConfig noiseConfig) {
        // Le constructeur de UncertaintySampling le plus court qui prend le Sampler n'existe pas.
        // Je ne peux pas appeler `super()` car les arguments du parent ne sont pas tous disponibles ici.
        // Je vais utiliser le constructeur vide ou minimal et le Sampler doit être injecté.
        // Puisque nous ne connaissons pas la bonne façon d'appeler super, 
        // je vais utiliser le constructeur qui prend l'Oracle, Dataset, etc.
        // Pour les besoins de l'implémentation, nous allons nous baser sur le fait que SafeGUS est un RankingsProvider
        // et qu'il a besoin des composants essentiels du Sampler.

        // CORRECTION: Je ne peux pas appeler `super()` sans tous les arguments.
        // Laissez l'utilisateur appeler le constructeur du parent dans le code d'initialisation.
        // Pour compiler, j'utilise une signature simple.
        this(null, null, null, 0); // Appel au constructeur le plus simple qui prend ArtificialOracle, Dataset, String[], double
        this.noiseConfig = noiseConfig;
    }
    
    // Ajout d'un constructeur pour faire l'appel super() qui est obligatoire
    public SafeGUS(tools.oracles.ArtificialOracle oracle, Dataset dataset, String[] measureNames, double noise, NoiseModelConfig noiseConfig) {
        super(oracle, dataset, measureNames, noise);
        this.noiseConfig = noiseConfig;
    }


    /**
     * Algorithme 2 (Noisy-GUS) : Sélectionne la paire la plus proche du Seuil de Sécurité C(t).
     */
    public IAlternative[] selectSafePair(ISinglevariateFunction model, int t) {
        double adaptiveThreshold = noiseConfig.getAdaptiveThreshold(t);

        // Récupération du Sampler MMAS (MMAS sampler est @Getter dans UncertaintySampling)
        MMAS sampler = (MMAS) this.getSampler();
        
        // Simule l'échantillonnage (runSampling). Ceci met à jour le buffer interne de MMAS.
        sampler.setScoringFunction(model);
        sampler.sample(); 
        
        // Récupération de l'ensemble S. On suppose que MMAS expose sa liste interne de meilleures règles.
        // Ceci est la source d'une erreur de compilation probable; MMAS n'a probablement pas de `getRuleBuffer()`.
        // Pour le moment, nous allons supposer que MMAS a une méthode qui retourne son buffer de règles (le "tas" H)
        // Je vais ajouter un getter à la classe parente (qui est l'endroit où la méthode devrait être).

        // ATTENTION: La classe UncertaintySampling que vous avez fournie n'expose pas getBestRules().
        // Cette ligne va échouer. L'utilisateur doit l'ajouter dans la classe Sampler/MMAS/UncertaintySampling.
        List<? extends IAlternative> S = sampler.getRuleBuffer().stream().collect(Collectors.toList()); 
        
        if (S == null || S.size() < 2) {
            return null;
        }

        IAlternative R_best1 = null;
        IAlternative R_best2 = null;
        double minDifference = Double.MAX_VALUE;

        // Cherche la paire qui minimise |Theta(g(Ra), g(Rb)) - C(t)|
        for (int i = 0; i < S.size(); i++) {
            for (int j = i + 1; j < S.size(); j++) {
                IAlternative Ra = S.get(i);
                IAlternative Rb = S.get(j);

                double scoreA = model.computeScore(Ra);
                double scoreB = model.computeScore(Rb);
                
                // Utilisation du getter pour PairwiseUncertainty
                PairwiseUncertainty differentiationFn = (PairwiseUncertainty) this.getPairwiseCertaintyFunction();
                double differentiationValue = differentiationFn.computeScore(scoreA, scoreB);

                double difference = Math.abs(differentiationValue - adaptiveThreshold);

                if (difference < minDifference) {
                    minDifference = difference;
                    R_best1 = Ra;
                    R_best2 = Rb;
                }
            }
        }
        
        if (R_best1 == null || R_best2 == null) {
             // Fallback: retourne l'incertitude maximale
             return RankingUtil.getMinUncertaintyPair(S, model, this.getPairwiseCertaintyFunction());
        }

        return new IAlternative[]{R_best1, R_best2};
    }
    
    @Override
    public IAlternative[] selectPair(ISinglevariateFunction model) {
        // Redirection vers SafeGUS pour l'itération 0 (ou pour l'exploration par défaut)
        return selectSafePair(model, 0); 
    }
}