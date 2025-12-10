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

            // --- 1. Chargement et Configuration ---
            String expDir = "src/test/resources/";
            String filename = "iris.dat";
            Set<String> consequents = new HashSet<>(Arrays.asList("Iris-setosa", "Iris-versicolor", "Iris-virginica"));
            
            Dataset dataset = new Dataset(filename, expDir, consequents);
            
            // DIAGNOSTIC 1 : Vérifier le dataset
            System.out.println("Dataset chargé : " + filename);
            System.out.println(" - Transactions : " + dataset.getNbTransactions());
            System.out.println(" - Items (Antecedents) : " + dataset.getNbAntecedentItems());
            if (dataset.getNbTransactions() == 0) {
                System.err.println("ERREUR CRITIQUE : Le dataset est vide ! Vérifiez le chemin du fichier.");
                return;
            }

            // Mesures utilisées (minuscules obligatoires)
            String[] measureNames = {"support", "confidence", "lift"}; 
            
            int maxIterations = 50; 
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

            // --- 3. Système de Monitoring ---
            String outputCsv = "results_noisy_letrid.csv";
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputCsv));
            writer.write("Iteration,Accuracy\n"); 

            // Génération du jeu de test
            List<DecisionRule> testRules = dataset.getRandomValidRules(200, 0.1, measureNames);
            
            // DIAGNOSTIC 2 : Vérifier les règles générées
            DecisionRule exemple = testRules.get(0);
            System.out.println("Exemple de règle générée : " + exemple);
            System.out.println(" - Fréquences : X=" + exemple.getFreqX() + ", Y=" + exemple.getFreqY() + ", Z=" + exemple.getFreqZ());
            if (exemple.getFreqZ() == 0) {
                System.err.println("ATTENTION : La règle exemple a une fréquence nulle. Les règles ne couvrent aucune transaction !");
            }

            // Observateur
            noisyAlgo.addObserver(evt -> {
                if ("step".equals(evt.getPropertyName())) {
                    NoisyLearnStep step = (NoisyLearnStep) evt.getNewValue();
                    ISinglevariateFunction currentModel = step.getCurrentScoreFunction();
                    int iteration = step.getIteration();

                    // Calcul de la précision
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

            // Vérité terrain
            int truth = oracle.compare(r1, r2); 
            
            // DIAGNOSTIC 3 : Ignorer les cas où l'oracle ne sait pas décider (scores égaux)
            if (truth == 0) {
                // L'oracle est indifférent, on ne compte pas cela comme une erreur ou une réussite
                // C'est souvent le cas si les règles sont identiques ou nulles
                continue; 
            }
            
            DecisionRule trueWinner = (truth > 0) ? r1 : r2;

            if (predictedWinner.equals(trueWinner)) {
                correct++;
            }
            total++;
        }
        
        if (total == 0) return 0.0;
        return (double) correct / total;
    }
}