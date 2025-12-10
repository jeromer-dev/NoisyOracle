package experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.ArrayList;

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

public class ExperimentNoisyLETRID {

    public static void main(String[] args) {
        try {
            System.out.println("=== Lancement de l'expérience Noisy-LETRID (Tic-Tac-Toe) ===");

            // --- 1. Chargement et Configuration ---
            String expDir = "src/test/resources/";
            
            // CHANGEMENT DE DATASET : Tic-Tac-Toe
            String filename = "tictactoe.dat";
            // Les classes dans ce dataset sont généralement "positive" et "negative"
            Set<String> consequents = new HashSet<>(Arrays.asList("positive", "negative"));
            
            Dataset dataset = new Dataset(filename, expDir, consequents);
            System.out.println("Dataset chargé : " + filename);
            System.out.println(" - Transactions : " + dataset.getNbTransactions());
            
            // Si le chargement échoue (0 transactions), on arrête
            if (dataset.getNbTransactions() == 0) {
                System.err.println("Erreur: Dataset vide ou mal lu.");
                return;
            }

            String[] measureNames = {"support", "confidence", "lift"}; 
            int maxIterations = 100; // On augmente un peu le budget pour ce dataset plus complexe
            double alpha = 0.5;

            // --- 2. Préparation des composants ---
            INoiseModel noiseModel = new ExponentialNoiseModel(5.0);
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), noiseModel, diffFunction);

            NoiseModelConfig config = new NoiseModelConfig(maxIterations, 0.8, 5.0);
            SafeGUS safeGus = new SafeGUS(noisyOracle, dataset, measureNames, config);
            CorrectionStrategy correctionStrategy = new CorrectionStrategy(diffFunction);
            
            NoisyIterativeRankingLearn noisyAlgo = new NoisyIterativeRankingLearn(
                maxIterations, alpha, safeGus, correctionStrategy, noisyOracle, config
            );

            // --- 3. Génération du Jeu de Test ---
            // On génère plus de règles pour avoir une évaluation plus fine
            List<DecisionRule> rawRules = dataset.getRandomValidRules(400, 0.1, measureNames);
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
            String outputCsv = "results_tictactoe.csv";
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
            
            // Affichage des poids appris pour analyse
            System.out.println("\n--- Poids Appris ---");
            if (finalParams.getWeights() != null) {
                double[] w = finalParams.getWeights();
                for(int i=0; i<w.length && i<measureNames.length; i++) {
                    System.out.println(measureNames[i] + ": " + String.format("%.4f", w[i]));
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
                int truth = oracle.compare(r1, r2); 
                if (truth == 0) continue; 
                
                double score1 = model.computeScore(r1);
                double score2 = model.computeScore(r2);
                DecisionRule predictedWinner = (score1 >= score2) ? r1 : r2;
                DecisionRule trueWinner = (truth < 0) ? r1 : r2; // -1 = r1 > r2
    
                if (predictedWinner.equals(trueWinner)) correct++;
                total++;
            } catch (Exception e) { continue; }
        }
        return (total == 0) ? 0.0 : (double) correct / total;
    }
}