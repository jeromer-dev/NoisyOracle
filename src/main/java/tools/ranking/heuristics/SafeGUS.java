package tools.ranking.heuristics;

import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.multivariate.CertaintyFunction; 
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.oracles.ArtificialOracle; 
import tools.utils.NoiseModelConfig; 
import tools.utils.RankingUtil;
import java.util.List;
import java.util.stream.Collectors;
import sampling.MMAS; 

public class SafeGUS extends UncertaintySampling {

    private final NoiseModelConfig noiseConfig;

    // Constructeur utilisant une surcharge existante dans UncertaintySampling
    public SafeGUS(ArtificialOracle oracle, Dataset dataset, String[] measureNames, NoiseModelConfig noiseConfig) {
        super(oracle, dataset, measureNames);
        this.noiseConfig = noiseConfig;
    }

    /**
     * Algorithme 2 (Noisy-GUS) : Sélectionne la paire la plus proche du Seuil de Sécurité C(t).
     */
    public IAlternative[] selectSafePair(ISinglevariateFunction model, int t) {
        double adaptiveThreshold = noiseConfig.getAdaptiveThreshold(t);

        // Récupération du Sampler MMAS via le getter (Corrigé Erreur #2)
        MMAS sampler = (MMAS) this.getSampler(); 
        
        // 1. Exécution de la logique SIMAS pour mettre à jour le buffer
        sampler.setScoringFunction(model);
        sampler.sample(); 
        
        // 2. Récupération de l'ensemble S via le getter (Corrigé Erreur #3)
        List<? extends IAlternative> S = sampler.getRuleBuffer().stream().collect(Collectors.toList()); 
        
        if (S == null || S.size() < 2) {
            return null;
        }

        IAlternative R_best1 = null;
        IAlternative R_best2 = null;
        double minDifference = Double.MAX_VALUE;

        // Récupération de la fonction de différenciation via le getter (Corrigé Erreur #2)
        CertaintyFunction differentiationFn = this.getPairwiseCertaintyFunction(); 

        // 3. Cherche la paire qui minimise |Theta(g(Ra), g(Rb)) - C(t)|
        for (int i = 0; i < S.size(); i++) {
            for (int j = i + 1; j < S.size(); j++) {
                IAlternative Ra = S.get(i);
                IAlternative Rb = S.get(j);

                double scoreA = model.computeScore(Ra);
                double scoreB = model.computeScore(Rb);
                
                double differentiationValue = differentiationFn.computeScore(scoreA, scoreB);

                double difference = Math.abs(differentiationValue - adaptiveThreshold);

                if (difference < minDifference) {
                    minDifference = difference;
                    R_best1 = Ra;
                    R_best2 = Rb;
                }
            }
        }
        
        // Fallback: retourne l'incertitude maximale si SafeGUS n'a pas trouvé de paire valide.
        if (R_best1 == null || R_best2 == null) {
             return RankingUtil.getMinUncertaintyPair(S, model, this.getPairwiseCertaintyFunction());
        }

        return new IAlternative[]{R_best1, R_best2};
    }
    
    // Implémentation du contrat d'interface/méthode (Corrigé Erreur #5)
    @Override
    public IAlternative[] selectPair(ISinglevariateFunction model) {
        return selectSafePair(model, 0); 
    }
}