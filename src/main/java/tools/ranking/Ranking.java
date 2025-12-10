package tools.ranking;

import lombok.Getter;
import lombok.ToString;
import java.util.Arrays;
import java.util.List;

/**
 * Represents the ranking of some objects by the user
 */
@ToString(of = { "ranking" })
public class Ranking<T> {

    /**
     * Positions of each index, ex: [1,2,0] means that obj of idx 2 is ranked first,
     * obj 0 second and obj 1 third
     */
    private @Getter int[] rankingPos;

    private @Getter Double[] scores;

    /** Objects that are ranked */
    private @Getter T[] objects;

    public Ranking(T[] objects, Double[] scores) {
        this.objects = objects;
        this.scores = scores;

        // Les parties commentées restent inchangées, nous nous basons sur la
        // fourniture d'objets déjà triés ou des scores pour la comparaison.
        // rankingPos = new int[ranking.length];
        // for (int i = 0; i < ranking.length; i++) {
        //     rankingPos[ranking[i]] = i;
        // }
    }

    /**
     * Prints the top 'k' ranked objects along with their ranks.
     *
     * @param k The number of top objects to print.
     */
    public void print(int k) {
        for (int i = 0; i < k; i++) {
            System.out.println((i + 1) + " : " + objects[i]);
        }
    }
    
    // CORRECTION Erreur #1: Ajout de getAlternatives()
    /**
     * Returns the ranked objects as a List.
     * @return List of ranked objects.
     */
    public List<T> getAlternatives() {
        return Arrays.asList(this.objects);
    }
    
    // CORRECTION Erreurs #2-5: Ajout de getRank(T alternative)
    /**
     * Retrieves the rank (index) of an object.
     * Comme les objets sont triés, l'index 0 est le rang 0 (meilleur).
     * @param alternative The object to find the rank for.
     * @return The zero-based rank (index) of the object, or -1 if not found.
     */
    public int getRank(T alternative) {
        for (int i = 0; i < this.objects.length; i++) {
            if (this.objects[i].equals(alternative)) {
                return i; 
            }
        }
        // Devrait retourner une position élevée si non trouvé, mais -1 est la pratique courante.
        return -1; 
    }
}