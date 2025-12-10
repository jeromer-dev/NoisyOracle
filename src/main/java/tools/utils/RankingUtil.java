package tools.utils;

import java.util.List;
import java.util.stream.IntStream;

import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.oracles.ArtificialOracle;
import tools.ranking.Ranking;
import tools.rules.DecisionRule;

/**
 * Utility class for working with rankings.
 */
public class RankingUtil {
        /**
         * Computes a ranking of alternatives based on the provided oracle comparator.
         * The ranking is determined by sorting the indices of alternatives according to
         * the oracle's comparison results.
         *
         * @param oracle The comparator used for determining the ranking order.
         * @param alternatives The array of alternatives to be ranked.
         * @return A Ranking object representing the computed ranking.
         */
        public static Ranking<IAlternative> computeRankingWithOracle(ArtificialOracle oracle, List<DecisionRule> rules,
                        IAlternative[] normalizedAlternatives) {

                int[] ranking = IntStream.range(0, rules.size())
                                .boxed()
                                // Sort the indices based on the scores computed by the oracle
                                .sorted((i, j) -> oracle.compare(rules.get(i), rules.get(j)))
                                .mapToInt(i -> i)
                                .toArray();

                // Initialize the alternativesArray based on the sorted rules
                IAlternative[] alternativesArray = new IAlternative[] { normalizedAlternatives[ranking[0]],
                                normalizedAlternatives[ranking[1]] };

                // Compute the oracle scores
                Double[] scores = new Double[] { oracle.computeScore(rules.get(ranking[0])),
                                oracle.computeScore(rules.get(ranking[1])) };

                // Return the final Ranking object
                return new Ranking<>(alternativesArray, scores);
        }

        public static Ranking<IAlternative> computeNoisyRankingWithOracle(ArtificialOracle oracle,
                        List<DecisionRule> rules, double noise) {

                // Create an array to hold the ranking indices
                int[] ranking = IntStream.range(0, rules.size())
                                .boxed()
                                // Sort the indices based on the scores computed by the oracle's scoring
                                // function.
                                .sorted((i, j) -> oracle.compareNoisy(rules.get(i), rules.get(j), noise))
                                .mapToInt(i -> i)
                                .toArray();

                IAlternative[] alternativesArray = rules.stream()
                                .sorted((rule1, rule2) -> oracle.compare(rule1, rule2))
                                .map(DecisionRule::getAlternative)
                                .toArray(IAlternative[]::new);

                // Compute the oracle scores
                Double[] scores = new Double[] { oracle.computeScore(rules.get(ranking[0])),
                                oracle.computeScore(rules.get(ranking[1])) };

                // TODO: give the normalized alternatives
                // Create and return a new Ranking object with the computed ranking
                return new Ranking<>(alternativesArray, scores);
        }
        
        // NOUVELLE METHODE AJOUTÉE POUR SAFEGUS (Résolution Erreur #4)
        /**
         * Retourne la paire d'alternatives ayant la plus petite incertitude (MinGaps original)
         */
        public static IAlternative[] getMinUncertaintyPair(
                List<? extends IAlternative> alternatives, 
                ISinglevariateFunction model, 
                CertaintyFunction differentiation) {
            
            IAlternative R1 = null;
            IAlternative R2 = null;
            double minUncertainty = Double.MAX_VALUE;

            for (int i = 0; i < alternatives.size(); i++) {
                for (int j = i + 1; j < alternatives.size(); j++) {
                    IAlternative Ra = alternatives.get(i);
                    IAlternative Rb = alternatives.get(j);

                    double scoreA = model.computeScore(Ra);
                    double scoreB = model.computeScore(Rb);
                    
                    // La fonction de différenciation calcule l'incertitude.
                    double uncertainty = differentiation.computeScore(scoreA, scoreB); 
                    
                    if (uncertainty < minUncertainty) {
                        minUncertainty = uncertainty;
                        R1 = Ra;
                        R2 = Rb;
                    }
                }
            }
            return new IAlternative[]{R1, R2};
        }
        // FIN DE LA NOUVELLE METHODE
}