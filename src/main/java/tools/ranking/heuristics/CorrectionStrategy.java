package tools.ranking.heuristics;

import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction; 
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.ranking.Ranking; 
import java.util.List;

/**
 * Algorithme 3 (CorrectionStrategy): Identification des incohérences (Exploitation).
 */
public class CorrectionStrategy {

    private final CertaintyFunction differentiationFunction;
    
    public CorrectionStrategy(CertaintyFunction differentiationFunction) {
        this.differentiationFunction = differentiationFunction;
    }

    /**
     * Parcourt l'historique U et retourne la paire la plus suspecte pour une revérification.
     * @param U Historique des préférences (List<Ranking<IAlternative>>).
     * @param model Le modèle actuel g_t.
     * @return Une paire suspecte (Ra, Rb) à revérifier, ou null.
     */
    public IAlternative[] findMostInconsistentPair(List<Ranking<IAlternative>> U, ISinglevariateFunction model) {
        
        double maxInconsistencyScore = -1.0;
        IAlternative[] bestCandidate = null;

        // Étape 3: for chaque tuple (Ri, Rj, y_label) in U
        for (Ranking<IAlternative> ranking : U) {
            
            // Utilisation du nouveau getAlternatives()
            List<IAlternative> alternatives = ranking.getAlternatives();
            if (alternatives.size() != 2) continue;
            
            IAlternative R1 = alternatives.get(0);
            IAlternative R2 = alternatives.get(1);
            
            // On détermine la préférence labellisée (y_label)
            IAlternative y_label;
            // Utilisation du nouveau getRank()
            if (ranking.getRank(R1) < ranking.getRank(R2)) {
                y_label = R1;
            } else if (ranking.getRank(R2) < ranking.getRank(R1)) {
                y_label = R2;
            } else {
                // Indifférence
                continue; 
            }
            
            double score1 = model.computeScore(R1);
            double score2 = model.computeScore(R2);
            
            // 4-8: Déterminer la prédiction du modèle (y_pred)
            IAlternative y_pred = null;
            if (score1 > score2) {
                y_pred = R1;
            } else if (score2 > score1) {
                y_pred = R2;
            } 
            
            // 9: if y_pred != y_label then
            if (y_pred != null && !y_pred.equals(y_label)) {
                // 10-11: Calcul de l'ampleur du désaccord (Certitude de différenciation)
                double differentiation = differentiationFunction.computeScore(score1, score2);
                
                // 12-14: Mise à jour du score maximal et du candidat
                if (differentiation > maxInconsistencyScore) {
                    maxInconsistencyScore = differentiation;
                    bestCandidate = new IAlternative[]{R1, R2};
                }
            }
        }
        
        // 18-20: return Ynew
        return bestCandidate; 
    }
}