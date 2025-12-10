package tools.oracles;

/**
 * Interface pour modéliser la probabilité d'erreur (bruit) de l'oracle
 * en fonction de la certitude de différenciation (Theta).
 */
public interface INoiseModel {
    
    /**
     * Calcule la probabilité que l'oracle fournisse une préférence erronée.
     * @param differentiation La certitude de différenciation Theta.
     * @return La probabilité d'erreur sigma.
     */
    double getErrorProbability(double differentiation);
}