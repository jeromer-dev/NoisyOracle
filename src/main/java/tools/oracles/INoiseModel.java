package tools.oracles;

/**
 * Interface pour modéliser la probabilité d'erreur (bruit) de l'oracle
 * en fonction de la certitude de différenciation (Theta).
 * sigma(Ri, Rj) = f(Theta(g(ri), g(rj)))
 */
public interface INoiseModel {
    
    /**
     * Calcule la probabilité que l'oracle fournisse une préférence erronée (sigma).
     * @param differentiation La certitude de différenciation Theta.
     * @return La probabilité d'erreur sigma (généralement entre 0.0 et 0.5).
     */
    double getErrorProbability(double differentiation);
}