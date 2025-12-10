package experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
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
            System.out.println("=== Lancement de l'expérience Noisy-LETRID ===");

            // 1. Chargement
            String expDir = "src/test/resources/";
            String filename = "iris.dat";
            
            // CORRECTION IMPORTANTE : Le fichier iris.dat contient des classes encodées en nombres (12, 13, 14)
            // et non "Iris-setosa". On adapte donc les conséquents.
            Set<String> consequents = new HashSet<>(Arrays.asList("12", "13", "14"));
            
            Dataset dataset = new Dataset(filename, expDir, consequents);
            System.out.println("Dataset chargé. Transactions : " + dataset.getNbTransactions());

            // CORRECTION : Mesures en minuscules obligatoires
            String[] measureNames = {"support", "confidence", "lift"}; 
            
            int maxIterations = 50; 
            double alpha = 0.5;

            // 2. Préparation
            INoiseModel noiseModel = new ExponentialNoiseModel(5.0);
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), noiseModel, diffFunction);

            NoiseModelConfig config = new NoiseModelConfig(maxIterations, 0.8, 5.0);
            SafeGUS safeGus = new SafeGUS(noisyOracle, dataset, measureNames, config);
            CorrectionStrategy correctionStrategy = new CorrectionStrategy(diffFunction);
            
            NoisyIterativeRankingLearn noisyAlgo = new NoisyIterativeRankingLearn(
                maxIterations, alpha, safeGus, correctionStrategy, noisyOracle, config
            );

            // 3. Monitoring
            String outputCsv = "results_noisy_letrid.csv";
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputCsv));
            writer.write("Iteration,Accuracy\n"); 

            // Génération du jeu de test (200 règles valides)
            List<DecisionRule> testRules = dataset.getRandomValidRules(200, 0.1, measureNames);
            if (testRules.isEmpty()) {
                System.err.println("ERREUR : Aucune règle de test n'a pu être générée. Vérifiez le dataset.");
                writer.close();
                return;
            }

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

            // 4. Lancement
            System.out.println("Début de l'apprentissage...");
            noisyAlgo.learn();
            
            writer.close();
            System.out.println("Expérience terminée.");

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

            double score1 = model.computeScore(r1);
            double score2 = model.computeScore(r2);
            DecisionRule predictedWinner = (score1 >= score2) ? r1 : r2;

            int truth = oracle.compare(r1, r2); 
            if (truth == 0) continue; // Ignore les égalités
            
            DecisionRule trueWinner = (truth > 0) ? r1 : r2;

            if (predictedWinner.equals(trueWinner)) correct++;
            total++;
        }
        return (total == 0) ? 0.0 : (double) correct / total;
    }
}