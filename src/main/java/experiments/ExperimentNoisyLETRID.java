package experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import tools.data.Dataset;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.oracles.ExponentialNoiseModel;
import tools.oracles.HumanLikeNoisyOracle;
import tools.oracles.INoiseModel;
import tools.ranking.heuristics.CorrectionStrategy;
import tools.ranking.heuristics.SafeGUS;
import tools.train.iterative.NoisyIterativeRankingLearn;
import tools.train.iterative.NoisyLearnStep;
import tools.utils.NoiseModelConfig;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;
import tools.rules.DecisionRule;
// import tools.rules.IRule; // Plus nécessaire ici

public class ExperimentNoisyLETRID {

    public static void main(String[] args) {
        try {
            System.out.println("=== Lancement de l'expérience Noisy-LETRID ===");

            // --- 1. Chargement et Configuration ---
            String expDir = "src/test/resources/";
            String filename = "iris.dat";
            // Codes des classes pour Iris
            Set<String> consequents = new HashSet<>(Arrays.asList("12", "13", "14"));
            
            Dataset dataset = new Dataset(filename, expDir, consequents);
            System.out.println("Dataset chargé. Transactions : " + dataset.getNbTransactions());

            String[] measureNames = {"support", "confidence", "lift"}; 
            int maxIterations = 50; 
            double alpha = 0.5;

            // --- 2. Préparation des composants ---
            INoiseModel noiseModel = new ExponentialNoiseModel(5.0);
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            
            // CORRECTION : Surcharge avec DecisionRule au lieu de IRule
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), noiseModel, diffFunction) {
                @Override
                public int compare(DecisionRule r1, DecisionRule r2) {
                    double s1 = this.computeScore(r1);
                    double s2 = this.computeScore(r2);
                    
                    if (s1 > s2) return -1; // r1 gagne
                    if (s2 > s1) return 1;  // r2 gagne
                    
                    // En cas d'égalité stricte du Chi2, on tranche par le Support (freqZ)
                    int f1 = r1.getFreqZ();
                    int f2 = r2.getFreqZ();
                    if (f1 > f2) return -1;
                    if (f2 > f1) return 1;
                    
                    // Si toujours égalité, on tranche arbitrairement
                    return (r1.hashCode() > r2.hashCode()) ? -1 : 1;
                }
            };

            NoiseModelConfig config = new NoiseModelConfig(maxIterations, 0.8, 5.0);
            SafeGUS safeGus = new SafeGUS(noisyOracle, dataset, measureNames, config);
            CorrectionStrategy correctionStrategy = new CorrectionStrategy(diffFunction);
            
            NoisyIterativeRankingLearn noisyAlgo = new NoisyIterativeRankingLearn(
                maxIterations, alpha, safeGus, correctionStrategy, noisyOracle, config
            );

            // --- 3. Génération du Jeu de Test ---
            List<DecisionRule> rawRules = dataset.getRandomValidRules(300, 0.1, measureNames);
            List<DecisionRule> testRules = new ArrayList<>();
            for (DecisionRule r : rawRules) {
                boolean isDuplicate = false;
                for (DecisionRule existing : testRules) {
                    if (existing.equals(r)) { isDuplicate = true; break; }
                }
                if (!isDuplicate) testRules.add(r);
                if (testRules.size() >= 200) break;
            }
            System.out.println("Règles de test uniques générées : " + testRules.size());

            // --- 4. Monitoring & Lancement ---
            String outputCsv = "results_noisy_letrid.csv";
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputCsv));
            writer.write("Iteration,Accuracy\n"); 

            noisyAlgo.addObserver(evt -> {
                if ("step".equals(evt.getPropertyName())) {
                    NoisyLearnStep step = (NoisyLearnStep) evt.getNewValue();
                    ISinglevariateFunction currentModel = step.getCurrentScoreFunction();
                    int iteration = step.getIteration();

                    double accuracy = computeAccuracy(currentModel, noisyOracle, testRules);
                    System.out.println(">>> Iteration " + iteration + " | Précision: " + String.format("%.2f", accuracy * 100) + "%");
                    
                    try {
                        writer.write(iteration + "," + accuracy + "\n");
                        writer.flush();
                    } catch (IOException e) { e.printStackTrace(); }
                }
            });

            System.out.println("Début de l'apprentissage...");
            FunctionParameters finalParams = noisyAlgo.learn();
            
            writer.close();
            System.out.println("Expérience terminée.");
            
            // Affichage des poids appris
            System.out.println("\nPoids appris :");
            if (finalParams.getWeights() != null) {
                for (int i = 0; i < finalParams.getWeights().length; i++) {
                    System.out.println(measureNames[i] + ": " + String.format("%.4f", finalParams.getWeights()[i]));
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double computeAccuracy(ISinglevariateFunction model, HumanLikeNoisyOracle oracle, List<DecisionRule> rules) {
        int correct = 0;
        int total = 0;

        for (int i = 0; i < rules.size() - 1; i += 2) {
            DecisionRule r1 = rules.get(i);
            DecisionRule r2 = rules.get(i+1);

            try {
                // On utilise notre oracle "Tie-Breaker" ici aussi
                int truth = oracle.compare(r1, r2); 
                
                double score1 = model.computeScore(r1);
                double score2 = model.computeScore(r2);
                DecisionRule predictedWinner = (score1 >= score2) ? r1 : r2;
                
                // -1 = r1 est mieux, 1 = r2 est mieux
                DecisionRule trueWinner = (truth < 0) ? r1 : r2;
    
                if (predictedWinner.equals(trueWinner)) correct++;
                total++;
                
            } catch (Exception e) { continue; }
        }
        return (total == 0) ? 0.0 : (double) correct / total;
    }
}