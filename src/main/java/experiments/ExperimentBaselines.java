package experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.oracles.ExponentialNoiseModel;
import tools.oracles.HumanLikeNoisyOracle;
import tools.oracles.INoiseModel;
import tools.ranking.Ranking;
import tools.utils.RandomUtil;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;
import tools.rules.DecisionRule;

public class ExperimentBaselines {

    // Paramètres globaux
    private static final int MAX_ITERATIONS = 50;
    private static final String FILENAME = "tictactoe.dat";
    private static final String[] MEASURES = {"support", "confidence", "lift"};
    
    public static void main(String[] args) {
        try {
            System.out.println("=== Lancement des Baselines ===");

            // 1. Chargement des données
            String expDir = "src/test/resources/";
            Set<String> consequents = new HashSet<>(Arrays.asList("positive", "negative", "class", "28", "29"));
            Dataset dataset = new Dataset(filenamePath(expDir, FILENAME), expDir, consequents);
            
            if (dataset.getNbTransactions() == 0) {
                System.err.println("Erreur: Dataset vide.");
                return;
            }
            System.out.println("Dataset chargé : " + dataset.getNbTransactions() + " transactions.");

            // 2. Préparation des Oracles
            INoiseModel noiseModel = new ExponentialNoiseModel(5.0);
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            
            // Oracle Bruité (Pour la Baseline Réaliste)
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), noiseModel, diffFunction) {
                @Override
                public int compare(DecisionRule r1, DecisionRule r2) {
                    return safeCompare(this, r1, r2);
                }
            };
            
            // Oracle Parfait (Pour la Borne Supérieure - Ideal)
            HumanLikeNoisyOracle perfectOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), noiseModel, diffFunction) {
                 @Override
                 public tools.alternatives.IAlternative getNoisyPreferredAlternative(tools.alternatives.IAlternative R1, tools.alternatives.IAlternative R2, ISinglevariateFunction model) {
                     // Pas de bruit ici : on retourne toujours le vrai gagnant
                     DecisionRule r1 = (DecisionRule) R1;
                     DecisionRule r2 = (DecisionRule) R2;
                     double s1 = this.computeScore(r1);
                     double s2 = this.computeScore(r2);
                     if (s1 > s2) return r1;
                     if (s2 > s1) return r2;
                     return (r1.hashCode() > r2.hashCode()) ? r1 : r2;
                 }
                 @Override
                 public int compare(DecisionRule r1, DecisionRule r2) {
                     return safeCompare(this, r1, r2);
                 }
            };

            // 3. Génération du Jeu de Test
            List<DecisionRule> testRules = generateTestSet(dataset);
            System.out.println("Jeu de test : " + testRules.size() + " règles.");

            // --- LANCEMENT SCENARIO 1 : BASELINE NOISY (GUS + Bruit) ---
            System.out.println("\n--- Scenario 1: Baseline (GUS + Bruit) ---");
            runExperiment(dataset, noisyOracle, testRules, "results_baseline_noisy.csv", true);

            // --- LANCEMENT SCENARIO 2 : IDEAL (GUS + Oracle Parfait) ---
            System.out.println("\n--- Scenario 2: Ideal (GUS + Parfait) ---");
            runExperiment(dataset, perfectOracle, testRules, "results_baseline_ideal.csv", false);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    // --- Moteur de l'expérience ---
    
    private static void runExperiment(Dataset dataset, HumanLikeNoisyOracle oracle, List<DecisionRule> testRules, String outputFile, boolean isNoisy) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        writer.write("Iteration,Accuracy\n");
        
        // Initialisation du modèle (Poids aléatoires normalisés)
        double[] weights = new double[MEASURES.length];
        double[] maxValues = new double[MEASURES.length];
        Arrays.fill(maxValues, 1.0);
        RandomUtil rng = new RandomUtil();
        double sum = 0;
        for(int i=0; i<weights.length; i++) { weights[i] = rng.nextDouble(); sum += weights[i]; }
        for(int i=0; i<weights.length; i++) weights[i] /= sum;
        
        List<Ranking<IAlternative>> history = new ArrayList<>();
        
        for (int t = 1; t <= MAX_ITERATIONS; t++) {
            // 1. Sélection GUS (Greedy Uncertainty Sampling)
            // On cherche la paire la plus incertaine (scores les plus proches)
            DecisionRule[] pair = selectMostUncertainPair(dataset, new NormalizedLinearFunction(weights, maxValues), rng);
            
            if (pair != null) {
                // Mise à jour de la normalisation
                updateMax(maxValues, pair[0]);
                updateMax(maxValues, pair[1]);
                
                // 2. Interrogation de l'Oracle
                // Note : Pour l'Active Learning classique, on utilise le modèle courant pour guider l'oracle (si besoin) 
                // mais ici l'oracle décide selon sa vérité (bruitée ou non)
                ISinglevariateFunction currentModel = new NormalizedLinearFunction(weights, maxValues);
                IAlternative preferred = oracle.getNoisyPreferredAlternative(pair[0], pair[1], currentModel);
                
                Ranking<IAlternative> ranking;
                if (preferred != null) {
                    IAlternative other = preferred.equals(pair[0]) ? pair[1] : pair[0];
                    ranking = new Ranking<>(new IAlternative[]{preferred, other}, new Double[]{1.0, 0.0});
                } else {
                    ranking = new Ranking<>(new IAlternative[]{pair[0], pair[1]}, new Double[]{0.5, 0.5});
                }
                history.add(ranking);
                
                // 3. Apprentissage (Mise à jour des poids)
                weights = doLearnStep(weights, maxValues, history);
            }
            
            // 4. Évaluation
            double acc = computeAccuracy(new NormalizedLinearFunction(weights, maxValues), oracle, testRules);
            writer.write(t + "," + acc + "\n");
            writer.flush();
            
            if (t % 10 == 0) System.out.println("Iter " + t + " : " + String.format("%.2f", acc * 100) + "%");
        }
        writer.close();
    }
    
    // --- Heuristique GUS (Sélection) ---
    
    private static DecisionRule[] selectMostUncertainPair(Dataset dataset, ISinglevariateFunction model, RandomUtil rng) {
        // Version simplifiée de GUS : on tire un batch aléatoire et on prend la paire la plus proche
        // Cela évite de dépendre des classes complexes SIMAS/UncertaintySampling du code de base
        List<DecisionRule> candidates = dataset.getRandomValidRules(50, 0.1, MEASURES);
        
        if (candidates.size() < 2) return null;
        
        DecisionRule bestR1 = null, bestR2 = null;
        double minDiff = Double.MAX_VALUE;
        
        for (int i = 0; i < candidates.size(); i++) {
            for (int j = i + 1; j < candidates.size(); j++) {
                double s1 = model.computeScore(candidates.get(i));
                double s2 = model.computeScore(candidates.get(j));
                double diff = Math.abs(s1 - s2);
                
                if (diff < minDiff) {
                    minDiff = diff;
                    bestR1 = candidates.get(i);
                    bestR2 = candidates.get(j);
                }
            }
        }
        return new DecisionRule[]{bestR1, bestR2};
    }

    // --- Moteur d'Apprentissage (Copie conforme de votre version stabilisée) ---

    private static double[] doLearnStep(double[] currentWeights, double[] maxValues, List<Ranking<IAlternative>> history) {
        double learningRate = 0.01; 
        int epochs = 100;
        int dim = currentWeights.length;
        double[] weights = Arrays.copyOf(currentWeights, dim); // Copie pour ne pas muter en place si échec
        
        NormalizedLinearFunction tempModel = new NormalizedLinearFunction(weights, maxValues);

        for (int e = 0; e < epochs; e++) {
            for (Ranking<IAlternative> rank : history) {
                IAlternative winner = rank.getObjects()[0];
                IAlternative loser = rank.getObjects()[1];
                Double[] scores = rank.getScores();
                if (scores != null && scores[0].equals(scores[1])) continue;

                double sW = tempModel.computeScore(winner);
                double sL = tempModel.computeScore(loser);
                
                double diff = sW - sL;
                double prob = 1.0 / (1.0 + Math.exp(-diff));
                double gradient = 1.0 - prob;

                double[] vecW = winner.getVector();
                double[] vecL = loser.getVector();

                for (int i = 0; i < dim; i++) {
                    double normW = vecW[i] / maxValues[i];
                    double normL = vecL[i] / maxValues[i];
                    weights[i] += learningRate * gradient * (normW - normL);
                    if (weights[i] < 0) weights[i] = 0;
                }
                tempModel = new NormalizedLinearFunction(weights, maxValues);
            }
        }
        
        double sum = 0;
        for (double w : weights) sum += w;
        if (sum > 0) for (int i = 0; i < dim; i++) weights[i] /= sum;
        
        return weights;
    }

    // --- Utilitaires ---

    private static String filenamePath(String dir, String name) { return name; } // Hack pour le constructeur Dataset si besoin

    private static List<DecisionRule> generateTestSet(Dataset dataset) {
        List<DecisionRule> raw = dataset.getRandomValidRules(400, 0.1, MEASURES);
        List<DecisionRule> unique = new ArrayList<>();
        for (DecisionRule r : raw) {
            boolean d = false;
            for (DecisionRule u : unique) if (u.equals(r)) d = true;
            if (!d) unique.add(r);
            if (unique.size() >= 200) break;
        }
        return unique;
    }

    private static void updateMax(double[] maxValues, IAlternative alt) {
        double[] vec = alt.getVector();
        for(int i=0; i<vec.length; i++) if(Math.abs(vec[i]) > maxValues[i]) maxValues[i] = Math.abs(vec[i]);
    }

    private static int safeCompare(HumanLikeNoisyOracle oracle, DecisionRule r1, DecisionRule r2) {
        double s1 = oracle.computeScore(r1);
        double s2 = oracle.computeScore(r2);
        if (s1 > s2) return -1;
        if (s2 > s1) return 1;
        if (r1.getFreqZ() > r2.getFreqZ()) return -1;
        if (r2.getFreqZ() > r1.getFreqZ()) return 1;
        return (r1.hashCode() > r2.hashCode()) ? -1 : 1;
    }

    private static double computeAccuracy(ISinglevariateFunction model, HumanLikeNoisyOracle oracle, List<DecisionRule> rules) {
        int correct = 0;
        int total = 0;
        for (int i = 0; i < rules.size() - 1; i += 2) {
            DecisionRule r1 = rules.get(i);
            DecisionRule r2 = rules.get(i+1);
            try {
                int truth = safeCompare(oracle, r1, r2);
                double s1 = model.computeScore(r1);
                double s2 = model.computeScore(r2);
                int predicted = (s1 >= s2) ? -1 : 1;
                if (predicted == truth) correct++;
                total++;
            } catch (Exception e) {}
        }
        return (total == 0) ? 0 : (double) correct / total;
    }

    private static class NormalizedLinearFunction extends LinearScoreFunction {
        private double[] w; private double[] m;
        public NormalizedLinearFunction(double[] weights, double[] max) { super(weights); this.w = weights; this.m = max; }
        @Override public double computeScore(IAlternative alt) {
            double s = 0; double[] v = alt.getVector();
            for(int i=0; i<v.length; i++) s += w[i] * (m[i]>0 ? v[i]/m[i] : 0);
            return s;
        }
        public double computeScore(DecisionRule r) { return computeScore((IAlternative)r); }
    }
}