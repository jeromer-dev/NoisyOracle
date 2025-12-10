package tools.ranking.heuristics;

import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction; 
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.ranking.Ranking; 
import java.util.List;

public class CorrectionStrategy {

    private final CertaintyFunction differentiationFunction;
    
    public CorrectionStrategy(CertaintyFunction differentiationFunction) {
        this.differentiationFunction = differentiationFunction;
    }

    public IAlternative[] findMostInconsistentPair(List<Ranking<IAlternative>> U, ISinglevariateFunction model) {
        
        double maxInconsistencyScore = -1.0;
        IAlternative[] bestCandidate = null;

        for (Ranking<IAlternative> ranking : U) {
            
            // CORRECTION: Utilise getObjects() qui existe dans la classe Ranking originale
            IAlternative[] objects = ranking.getObjects();
            if (objects == null || objects.length != 2) continue;
            
            IAlternative R1 = objects[0];
            IAlternative R2 = objects[1];
            
            // CORRECTION: Logique de rang basée sur l'index (0 = 1er/préféré, 1 = 2ème)
            IAlternative y_label = objects[0]; // Par définition dans un Ranking, le premier est le mieux classé
            
            // Si les scores sont égaux (indifférence), on ignore
            // On vérifie si un score est disponible dans l'objet Ranking
            Double[] scores = ranking.getScores();
            if (scores != null && scores.length >= 2 && scores[0].equals(scores[1])) {
                continue; // Indifférence
            }

            double score1 = model.computeScore(R1);
            double score2 = model.computeScore(R2);
            
            IAlternative y_pred = null;
            if (score1 > score2) {
                y_pred = R1;
            } else if (score2 > score1) {
                y_pred = R2;
            } 
            
            // Si le modèle prédit une préférence stricte différente du label
            if (y_pred != null && !y_pred.equals(y_label)) {
                double differentiation = differentiationFunction.computeScore(score1, score2);
                
                if (differentiation > maxInconsistencyScore) {
                    maxInconsistencyScore = differentiation;
                    bestCandidate = new IAlternative[]{R1, R2};
                }
            }
        }
        
        return bestCandidate; 
    }
}