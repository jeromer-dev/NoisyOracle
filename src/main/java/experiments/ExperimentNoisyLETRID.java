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
import tools.utils.RandomUtil; // Import nécessaire pour le Random
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;
import tools.rules.DecisionRule;

public class ExperimentNoisyLETRID {

    public static void main(String[] args) {
        try {
            System.out.println("=== Lancement de l'expérience Noisy-LETRID (Tic-Tac-Toe) ===");

            // 1. Configuration
            String expDir = "src/test/resources/";
            String filename = "tictactoe.dat";
            Set<String> consequents = new HashSet<>(Arrays.asList("28", "29", "positive", "negative", "class"));
            
            Dataset dataset = new Dataset(filename, expDir, consequents);
            System.out.println("Dataset chargé. Transactions : " + dataset.getNbTransactions());
            
            if (dataset.getNbTransactions() == 0) return;

            String[] measureNames = {"support", "confidence", "lift"}; 
            int maxIterations = 50; 
            double alpha = 0.5;

            // 2. Oracle avec Départage Forcé (Tie-Breaking)
            INoiseModel noiseModel = new ExponentialNoiseModel(5.0);
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            
            // CORRECTION CRITIQUE : On surcharge getNoisyPreferredAlternative pour empêcher l'indifférence
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(dataset.getNbTransactions(), noiseModel, diffFunction) {
                private final RandomUtil rng = new RandomUtil();

                @Override
                public tools.alternatives.IAlternative getNoisyPreferredAlternative(tools.alternatives.IAlternative R1, tools.alternatives.IAlternative R2, ISinglevariateFunction model) {
                    DecisionRule r1 = (DecisionRule) R1;
                    DecisionRule r2 = (DecisionRule) R2;

                    // A. Calcul de la VRAIE préférence (sans bruit)
                    double s1 = this.computeScore(r1); // Score Chi2
                    double s2 = this.computeScore(r2);
                    
                    tools.alternatives.IAlternative truePreferred;
                    
                    if (s1 > s2) truePreferred = r1;
                    else if (s2 > s1) truePreferred = r2;
                    else {
                        // ÉGALITÉ CHI2 -> Départage par le Support
                        if (r1.getFreqZ() > r2.getFreqZ()) truePreferred = r1;
                        else if (r2.getFreqZ() > r1.getFreqZ()) truePreferred = r2;
                        // ÉGALITÉ SUPPORT -> Départage arbitraire (Hashcode)
                        else truePreferred = (r1.hashCode() > r2.hashCode()) ? r1 : r2;
                    }

                    // B. Ajout du BRUIT (Simulation erreur humaine)
                    double m1 = model.computeScore(R1);
                    double m2 = model.computeScore(R2);
                    double diff = diffFunction.computeScore(m1, m2);
                    double probError = noiseModel.getErrorProbability(diff);

                    if (rng.nextDouble() < probError) {
                        return truePreferred.equals(R1) ? R2 : R1; // Inversion due à l'erreur
                    }
                    return truePreferred;
                }
                
                // On surcharge aussi compare pour le calcul de précision (Test set)
                @Override
                public int compare(DecisionRule r1, DecisionRule r2) {
                    double s1 = this.computeScore(r1);
                    double s2 = this.computeScore(r2);
                    if (s1 > s2) return -1;
                    if (s2 > s1) return 1;
                    // Départage identique pour le test
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

            // 3. Monitoring
            String outputCsv = "results_noisy_letrid.csv";
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputCsv));
            writer.write("Iteration,Accuracy\n"); 

            // Génération large puis filtrage pour avoir de la diversité
            List<DecisionRule> rawRules = dataset.getRandomValidRules(500, 0.1, measureNames);
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
            try {
                // L'oracle surchargé renvoie toujours -1 ou 1 (jamais 0)
                int truth = oracle.compare(r1, r2);
                
                double score1 = model.computeScore(r1);
                double score2 = model.computeScore(r2);
                DecisionRule predictedWinner = (score1 >= score2) ? r1 : r2;
                DecisionRule trueWinner = (truth < 0) ? r1 : r2;
    
                if (predictedWinner.equals(trueWinner)) correct++;
                total++;
            } catch (Exception e) { continue; }
        }
        return (total == 0) ? 0.0 : (double) correct / total;
    }
}