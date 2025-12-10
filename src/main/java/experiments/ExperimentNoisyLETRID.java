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

public class ExperimentNoisyLETRID {

    public static void main(String[] args) {
        try {
            System.out.println("=== Lancement de l'expérience Noisy-LETRID (Tic-Tac-Toe) ===");

            // --- 1. Configuration pour Tic-Tac-Toe ---
            String expDir = "src/test/resources/";
            String filename = "tictactoe.dat";
            
            // Dans tictactoe.dat, les classes sont les derniers items. 
            // On suppose ici que ce sont "positive" et "negative" ou des IDs.
            // Pour être sûr, on inclut les IDs vus dans vos fichiers (28, 29) et les noms classiques.
            Set<String> consequents = new HashSet<>(Arrays.asList("positive", "negative", "28", "29", "class"));
            
            Dataset dataset = new Dataset(filename, expDir, consequents);
            System.out.println("Dataset chargé : " + filename);
            System.out.println(" - Transactions : " + dataset.getNbTransactions());
            
            if (dataset.getNbTransactions() == 0) {
                System.err.println("ERREUR : Dataset vide ! Vérifiez le fichier.");
                return;
            }

            String[] measureNames = {"support", "confidence", "lift"}; 
            int maxIterations = 50; 
            double alpha = 0.5;

            // --- 2. Préparation des composants ---
            INoiseModel noiseModel = new ExponentialNoiseModel(5.0);
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            
            // Oracle avec décision forcée (Tie-Breaker) pour éviter les indécisions
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), noiseModel, diffFunction) {
                @Override
                public int compare(DecisionRule r1, DecisionRule r2) {
                    double s1 = this.computeScore(r1);
                    double s2 = this.computeScore(r2);
                    
                    if (s1 > s2) return -1;
                    if (s2 > s1) return 1;
                    
                    // En cas d'égalité, on départage par le Support
                    if (r1.getFreqZ() > r2.getFreqZ()) return -1;
                    if (r2.getFreqZ() > r1.getFreqZ()) return 1;
                    
                    return (r1.hashCode() > r2.hashCode()) ? -1 : 1;
                }
            };

            NoiseModelConfig config = new NoiseModelConfig(maxIterations, 0.8, 5.0);
            SafeGUS safeGus = new SafeGUS(noisyOracle, dataset, measureNames, config);
            CorrectionStrategy correctionStrategy = new CorrectionStrategy(diffFunction);
            
            NoisyIterativeRankingLearn noisyAlgo = new NoisyIterativeRankingLearn(
                maxIterations, alpha, safeGus, correctionStrategy, noisyOracle, config
            );

            // --- 3. Jeu de Test & Logs ---
            String outputCsv = "results_noisy_letrid.csv";
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputCsv));
            writer.write("Iteration,Accuracy\n"); 

            // On génère assez de règles pour avoir un bon test
            List<DecisionRule> rawRules = dataset.getRandomValidRules(400, 0.1, measureNames);
            List<DecisionRule> testRules = new ArrayList<>();
            // Filtrage des doublons
            for (DecisionRule r : rawRules) {
                boolean isDuplicate = false;
                for (DecisionRule ex : testRules) {
                    if (ex.equals(r)) { isDuplicate = true; break; }
                }
                if (!isDuplicate) testRules.add(r);
                if (testRules.size() >= 200) break;
            }
            
            if (testRules.isEmpty()) {
                System.err.println("ERREUR : Impossible de générer des règles de test valides.");
                writer.close();
                return;
            }
            System.out.println("Règles de test uniques générées : " + testRules.size());

            // Abonnement pour afficher la progression
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

            // --- 4. Lancement ---
            System.out.println("Début de l'apprentissage...");
            FunctionParameters finalParams = noisyAlgo.learn();
            writer.close();
            
            System.out.println("Expérience terminée.");
            System.out.println("Poids finaux appris :");
            if (finalParams.getWeights() != null) {
                for(int i=0; i<finalParams.getWeights().length; i++) {
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
                int truth = oracle.compare(r1, r2); 
                
                double score1 = model.computeScore(r1);
                double score2 = model.computeScore(r2);
                DecisionRule predictedWinner = (score1 >= score2) ? r1 : r2;
                
                // -1 signifie que r1 est préféré par l'oracle
                DecisionRule trueWinner = (truth < 0) ? r1 : r2;
    
                if (predictedWinner.equals(trueWinner)) correct++;
                total++;
                
            } catch (Exception e) { continue; }
        }
        return (total == 0) ? 0.0 : (double) correct / total;
    }
}