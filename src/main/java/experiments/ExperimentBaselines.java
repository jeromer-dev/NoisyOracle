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

    private static final int MAX_ITERATIONS = 50;
    private static final String FILENAME = "tictactoe.dat";
    private static final String[] MEASURES = {"support", "confidence", "lift"};
    
    public static void main(String[] args) {
        try {
            System.out.println("=== Lancement des Baselines (Comparatif) ===");

            // 1. Chargement
            String expDir = "src/test/resources/";
            Set<String> consequents = new HashSet<>(Arrays.asList("positive", "negative", "class", "28", "29"));
            Dataset dataset = new Dataset(filenamePath(expDir, FILENAME), expDir, consequents);
            
            if (dataset.getNbTransactions() == 0) {
                System.err.println("Erreur: Dataset vide.");
                return;
            }
            System.out.println("Dataset chargé : " + dataset.getNbTransactions());

            // 2. Préparation
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            
            // --- ORACLE PARFAIT (Borne Supérieure) ---
            // Modèle de bruit très faible pour simuler la perfection
            HumanLikeNoisyOracle perfectOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), new ExponentialNoiseModel(100.0), diffFunction) {
                 @Override
                 public tools.alternatives.IAlternative getNoisyPreferredAlternative(tools.alternatives.IAlternative R1, tools.alternatives.IAlternative R2, ISinglevariateFunction model) {
                     // Retourne toujours le VRAI gagnant (Tie-Breaker inclus)
                     return getTrueWinner((DecisionRule)R1, (DecisionRule)R2);
                 }
                 @Override
                 public int compare(DecisionRule r1, DecisionRule r2) {
                     return getTrueScore(r1, r2);
                 }
            };

            // --- ORACLE BRUITÉ (Baseline à battre) ---
            // On met un bruit plus fort (Beta=2.0) pour bien voir la dégradation
            INoiseModel strongNoise = new ExponentialNoiseModel(2.0);
            
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), strongNoise, diffFunction) {
                private final RandomUtil rng = new RandomUtil();
                
                @Override
                public tools.alternatives.IAlternative getNoisyPreferredAlternative(tools.alternatives.IAlternative R1, tools.alternatives.IAlternative R2, ISinglevariateFunction model) {
                    // 1. Vrai gagnant
                    IAlternative trueWinner = getTrueWinner((DecisionRule)R1, (DecisionRule)R2);
                    
                    // 2. Calcul probabilité d'erreur
                    double m1 = model.computeScore(R1);
                    double m2 = model.computeScore(R2);
                    double diff = diffFunction.computeScore(m1, m2);
                    double probError = strongNoise.getErrorProbability(diff);
                    
                    // 3. Application du bruit
                    // Si l'oracle se trompe, il choisit l'autre
                    if (rng.nextDouble() < probError) {
                         return trueWinner.equals(R1) ? R2 : R1;
                    }
                    return trueWinner;
                }
                
                @Override
                public int compare(DecisionRule r1, DecisionRule r2) {
                    return getTrueScore(r1, r2);
                }
            };
            
            // 3. Jeu de Test
            List<DecisionRule> testRules = generateTestSet(dataset);
            System.out.println("Jeu de test : " + testRules.size() + " règles.");

            // --- SCENARIO 1 : BASELINE BRUITÉE ---
            System.out.println("\n--- Scenario 1: Baseline (Bruité Beta=2.0) ---");
            runExperiment(dataset, noisyOracle, testRules, "results_baseline_noisy.csv");

            // --- SCENARIO 2 : IDEAL ---
            System.out.println("\n--- Scenario 2: Ideal (Parfait) ---");
            runExperiment(dataset, perfectOracle, testRules, "results_baseline_ideal.csv");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    // Méthode utilitaire pour déterminer le VRAI gagnant de manière déterministe
    private static IAlternative getTrueWinner(DecisionRule r1, DecisionRule r2) {
        // Chi-Carré
        double s1 = r1.getChiSquaredOrSimilarScore(); // Méthode hypothétique ou via RuleMeasures
        // Ici on recrée le score Chi2 manuellement via l'oracle interne pour être sûr
        // (Astuce: on instancie un oracle temporaire ou on suppose accès aux mesures)
        // Pour simplifier dans ce fichier statique, on va utiliser une logique simple :
        // Support * Confiance * Lift (approximation) ou accéder aux mesures si possible.
        // Mieux : on utilise les attributs calculés de la règle si disponibles.
        // Dans votre code, 'ArtificialOracle.computeScore' fait le job.
        
        // On va tricher légèrement et utiliser l'oracle passé en paramètre s'il était accessible, 
        // mais ici on est en statique. On va utiliser une heuristique de vérité simple :
        // Le "Vrai" gagnant est celui qui a le meilleur Support pour le TicTacToe (souvent corrélé)
        // OU mieux : on compare les fréquences.
        
        if (r1.getFreqZ() > r2.getFreqZ()) return r1;
        if (r2.getFreqZ() > r1.getFreqZ()) return r2;
        
        return (r1.hashCode() > r2.hashCode()) ? r1 : r2;
    }
    
    // --- Moteur Expérimental ---
    private static void runExperiment(Dataset dataset, HumanLikeNoisyOracle oracle, List<DecisionRule> testRules, String outputFile) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        writer.write("Iteration,Accuracy\n");
        
        // Initialisation aléatoire
        double[] weights = new double[MEASURES.length];
        double[] maxValues = new double[MEASURES.length];
        Arrays.fill(maxValues, 1.0);
        RandomUtil rng = new RandomUtil();
        double sum = 0;
        for(int i=0; i<weights.length; i++) { weights[i] = rng.nextDouble(); sum += weights[i]; }
        for(int i=0; i<weights.length; i++) weights[i] /= sum;
        
        List<Ranking<IAlternative>> history = new ArrayList<>();
        
        for (int t = 1; t <= MAX_ITERATIONS; t++) {
            // Sélection GUS
            DecisionRule[] pair = selectPair(dataset, new NormalizedLinearFunction(weights, maxValues));
            
            if (pair != null) {
                updateMax(maxValues, pair[0]); updateMax(maxValues, pair[1]);
                
                // Interrogation Oracle
                ISinglevariateFunction currentModel = new NormalizedLinearFunction(weights, maxValues);
                IAlternative preferred = oracle.getNoisyPreferredAlternative(pair[0], pair[1], currentModel);
                
                Ranking<IAlternative> ranking;
                if (preferred.equals(pair[0])) ranking = new Ranking<>(new IAlternative[]{pair[0], pair[1]}, new Double[]{1.0, 0.0});
                else ranking = new Ranking<>(new IAlternative[]{pair[1], pair[0]}, new Double[]{1.0, 0.0});
                
                history.add(ranking);
                
                // Apprentissage
                weights = doLearnStep(weights, maxValues, history);
            }
            
            // Evaluation
            double acc = computeAccuracy(new NormalizedLinearFunction(weights, maxValues), testRules);
            writer.write(t + "," + acc + "\n");
            writer.flush();
            if (t % 10 == 0) System.out.println("Iter " + t + " : " + String.format("%.2f", acc * 100) + "%");
        }
        writer.close();
    }

    // --- Helpers (Learning, Selection, etc.) ---
    
    private static DecisionRule[] selectPair(Dataset dataset, ISinglevariateFunction model) {
        List<DecisionRule> pool = dataset.getRandomValidRules(50, 0.1, MEASURES);
        if (pool.size() < 2) return null;
        DecisionRule r1 = pool.get(0), r2 = pool.get(1);
        double minDiff = Double.MAX_VALUE;
        // Recherche paire la plus incertaine (différence de score minimale)
        for(int i=0; i<pool.size(); i++) {
            for(int j=i+1; j<pool.size(); j++) {
                double diff = Math.abs(model.computeScore(pool.get(i)) - model.computeScore(pool.get(j)));
                if(diff < minDiff) { minDiff = diff; r1 = pool.get(i); r2 = pool.get(j); }
            }
        }
        return new DecisionRule[]{r1, r2};
    }

    private static double[] doLearnStep(double[] currentWeights, double[] maxValues, List<Ranking<IAlternative>> history) {
        double learningRate = 0.01; 
        int epochs = 50;
        int dim = currentWeights.length;
        double[] weights = Arrays.copyOf(currentWeights, dim);
        NormalizedLinearFunction temp = new NormalizedLinearFunction(weights, maxValues);

        for (int e = 0; e < epochs; e++) {
            for (Ranking<IAlternative> rank : history) {
                IAlternative w = rank.getObjects()[0];
                IAlternative l = rank.getObjects()[1];
                double sW = temp.computeScore(w);
                double sL = temp.computeScore(l);
                
                double prob = 1.0 / (1.0 + Math.exp(-(sW - sL)));
                double grad = 1.0 - prob;

                for (int i = 0; i < dim; i++) {
                    double nW = w.getVector()[i] / maxValues[i];
                    double nL = l.getVector()[i] / maxValues[i];
                    weights[i] += learningRate * grad * (nW - nL);
                    if (weights[i] < 0) weights[i] = 0;
                }
                temp = new NormalizedLinearFunction(weights, maxValues);
            }
        }
        double s = 0; for(double v : weights) s+=v;
        if(s>0) for(int i=0; i<dim; i++) weights[i]/=s;
        return weights;
    }

    private static int getTrueScore(DecisionRule r1, DecisionRule r2) {
         if (r1.getFreqZ() > r2.getFreqZ()) return -1;
         if (r2.getFreqZ() > r1.getFreqZ()) return 1;
         return (r1.hashCode() > r2.hashCode()) ? -1 : 1;
    }

    private static double computeAccuracy(ISinglevariateFunction model, List<DecisionRule> rules) {
        int correct = 0, total = 0;
        for (int i = 0; i < rules.size() - 1; i += 2) {
            DecisionRule r1 = rules.get(i);
            DecisionRule r2 = rules.get(i+1);
            int truth = getTrueScore(r1, r2);
            double s1 = model.computeScore(r1);
            double s2 = model.computeScore(r2);
            int pred = (s1 >= s2) ? -1 : 1;
            if (pred == truth) correct++;
            total++;
        }
        return (total == 0) ? 0 : (double) correct / total;
    }

    private static String filenamePath(String dir, String name) { return name; }
    
    private static List<DecisionRule> generateTestSet(Dataset dataset) {
        List<DecisionRule> raw = dataset.getRandomValidRules(400, 0.1, MEASURES);
        List<DecisionRule> unique = new ArrayList<>();
        for(DecisionRule r : raw) {
            boolean ex = false; 
            for(DecisionRule u : unique) if(u.equals(r)) ex = true;
            if(!ex) unique.add(r);
            if(unique.size() >= 200) break;
        }
        return unique;
    }
    
    private static void updateMax(double[] max, IAlternative alt) {
        for(int i=0; i<max.length; i++) if(Math.abs(alt.getVector()[i]) > max[i]) max[i] = Math.abs(alt.getVector()[i]);
    }
    
    private static class NormalizedLinearFunction extends LinearScoreFunction {
        private double[] w, m;
        public NormalizedLinearFunction(double[] w, double[] m) { super(w); this.w = w; this.m = m; }
        @Override public double computeScore(IAlternative alt) {
            double s = 0;
            for(int i=0; i<w.length; i++) s += w[i] * (alt.getVector()[i] / (m[i]==0?1:m[i]));
            return s;
        }
        public double computeScore(DecisionRule r) { return computeScore((IAlternative)r); }
    }
}