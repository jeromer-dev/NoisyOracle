package experiments;

import java.util.ArrayList;
import java.util.List;

import tools.data.Dataset;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.metrics.ExperimentLogger;
import tools.oracles.ExponentialNoiseModel;
import tools.oracles.HumanLikeNoisyOracle;
import tools.oracles.INoiseModel;
import tools.ranking.heuristics.CorrectionStrategy;
import tools.ranking.heuristics.SafeGUS;
import tools.train.iterative.NoisyIterativeRankingLearn;
import tools.utils.NoiseModelConfig;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;

public class ExperimentNoisyLETRID {

    public static void main(String[] args) {
        try {
            System.out.println("=== Lancement de l'expérience Noisy-LETRID ===");

            // 1. Chargement du Dataset
            // Assurez-vous que le chemin est correct selon votre structure de projet
            String datasetPath = "src/test/resources/iris.dat"; 
            Dataset dataset = new Dataset(datasetPath);
            System.out.println("Dataset chargé : " + datasetPath);

            // 2. Configuration Commune
            int maxIterations = 50; // Budget L
            String[] measureNames = dataset.getMeasureNames();
            
            // --- Configuration du Modèle de Bruit (L'Humain Simulé) ---
            // On simule un expert avec une sensibilité Beta = 5.0 (Bruit Exponentiel)
            INoiseModel noiseModel = new ExponentialNoiseModel(5.0);
            
            // Fonction de différenciation pour calculer Theta (incertitude)
            PairwiseUncertainty diffFunction = new PairwiseUncertainty("ScoreDiff", new ScoreDifference(new LinearScoreFunction()));
            
            // L'Oracle Bruité (Vérité Terrain = ChiSquared sur le dataset)
            HumanLikeNoisyOracle noisyOracle = new HumanLikeNoisyOracle(100, noiseModel, diffFunction);
            // On entraîne l'oracle sur les données pour qu'il ait une "vérité" à bruiter
            noisyOracle.learn(dataset, measureNames);

            // --- Configuration de Noisy-LETRID ---
            NoiseModelConfig config = new NoiseModelConfig(maxIterations, 0.8, 5.0);
            
            // Composants de Noisy-LETRID
            SafeGUS safeGus = new SafeGUS(noisyOracle, dataset, measureNames, config);
            CorrectionStrategy correctionStrategy = new CorrectionStrategy(diffFunction); // Utilise la même fonction de diff
            
            // Paramètre Alpha (Compromis Exploration/Exploitation)
            double alpha = 0.5; 

            System.out.println("Configuration : Alpha=" + alpha + ", Budget=" + maxIterations);

            // 3. Instanciation de l'Algorithme
            NoisyIterativeRankingLearn noisyAlgo = new NoisyIterativeRankingLearn(
                maxIterations, 
                alpha, 
                safeGus, 
                correctionStrategy, 
                noisyOracle, 
                config
            );

            // Ajout d'un logger pour voir la progression (si supporté)
            // noisyAlgo.addObserver(new ExperimentLogger()); 

            // 4. Lancement de l'apprentissage
            System.out.println("Début de l'apprentissage...");
            long startTime = System.currentTimeMillis();
            
            FunctionParameters params = noisyAlgo.learn();
            
            long endTime = System.currentTimeMillis();
            System.out.println("Apprentissage terminé en " + (endTime - startTime) + "ms.");
            
            // 5. Affichage des résultats
            System.out.println("Poids finaux appris :");
            System.out.println(params);

            // TODO: Ici, vous pourriez comparer 'params' avec les vrais poids de l'oracle (si connus)
            // ou calculer un score de Kendall Tau sur un jeu de test.

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}