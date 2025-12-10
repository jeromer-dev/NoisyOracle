package experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.oracles.ExponentialNoiseModel;
import tools.oracles.HumanLikeNoisyOracle;
import tools.oracles.INoiseModel;
import tools.ranking.heuristics.CorrectionStrategy;
import tools.ranking.heuristics.SafeGUS;
import tools.train.LearnStep;
import tools.train.iterative.NoisyIterativeRankingLearn;
import tools.utils.NoiseModelConfig;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;
import tools.rules.DecisionRule;

public class ExperimentNoisyLETRID {

    public static void main(String[] args) {
        try {
            System.out.println("=== Lancement de l'expérience Noisy-LETRID ===");

            // --- 1. Chargement et Configuration ---
            String expDir = "src/test/resources/";
            String filename = "iris.dat";
            Set<String> consequents = new HashSet<>(Arrays.asList("Iris-setosa", "Iris-versicolor", "Iris-virginica"));
            
            Dataset dataset = new Dataset(filename, expDir, consequents);
            String[] measureNames = {"Support", "Confidence", "GrowthRate"}; 
            int maxIterations = 50; 
            double alpha = 0.5;

            // --- 2. Préparation des composants ---
            INoiseModel noiseModel = new ExponentialNoiseModel(5.0);
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            
            // Oracle Bruité (L'humain)
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), noiseModel, diffFunction);

            // Algorithme Noisy-LETRID
            NoiseModelConfig config = new NoiseModelConfig(maxIterations, 0.8, 5.0);
            SafeGUS safeGus = new SafeGUS(noisyOracle, dataset, measureNames, config);
            CorrectionStrategy correctionStrategy = new CorrectionStrategy(diffFunction);
            
            NoisyIterativeRankingLearn noisyAlgo = new NoisyIterativeRankingLearn(
                maxIterations, alpha, safeGus, correctionStrategy, noisyOracle, config
            );

            // --- 3. Système de Monitoring (Log dans CSV) ---
            String outputCsv = "results_noisy_letrid.csv";
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputCsv));
            writer.write("Iteration,Accuracy\n"); // En-tête du CSV

            // On génère un jeu de test fixe de 100 paires pour évaluer la précision à chaque tour
            List<DecisionRule> testRules = dataset.getRandomValidRules(200, 0.1, measureNames);

            // On s'abonne aux événements de l'algorithme
            noisyAlgo.addObserver(evt -> {
                if ("step".equals(evt.getPropertyName())) {
                    LearnStep step = (LearnStep) evt.getNewValue();
                    ISinglevariateFunction currentModel = step.getCurrentScoreFunction();
                    int iteration = step.getIteration();

                    // Calcul de la précision sur le jeu de test
                    double accuracy = computeAccuracy(currentModel, noisyOracle, testRules);
                    
                    System.out.println(">>> Iteration " + iteration + " | Précision: " + String.format("%.2f", accuracy * 100) + "%");
                    
                    try {
                        writer.write(iteration + "," + accuracy + "\n");
                        writer.flush();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });

            // --- 4. Lancement ---
            System.out.println("Début de l'apprentissage... Résultats dans " + outputCsv);
            noisyAlgo.learn();
            
            writer.close();
            System.out.println("Expérience terminée.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Calcule la précision du modèle actuel par rapport à la vérité terrain (Oracle sans bruit).
     */
    private static double computeAccuracy(ISinglevariateFunction model, HumanLikeNoisyOracle oracle, List<DecisionRule> rules) {
        int correct = 0;
        int total = 0;

        // On compare les règles 2 à 2 dans la liste de test
        for (int i = 0; i < rules.size() - 1; i += 2) {
            DecisionRule r1 = rules.get(i);
            DecisionRule r2 = rules.get(i+1);

            double score1 = model.computeScore(r1);
            double score2 = model.computeScore(r2);

            // Prédiction du modèle (qui gagne ?)
            DecisionRule predictedWinner = (score1 >= score2) ? r1 : r2;

            // Vérité terrain (Oracle de base, SANS le bruit ajouté par getNoisyPreferred...)
            // On utilise la méthode compare() héritée qui donne la vraie préférence mathématique
            int truth = oracle.compare(r1, r2); 
            DecisionRule trueWinner = (truth >= 0) ? r1 : r2;

            if (predictedWinner.equals(trueWinner)) {
                correct++;
            }
            total++;
        }
        return (double) correct / total;
    }
}