package tools.oracles;

/**
 * Modèle de bruit Exponentiel.
 * La probabilité d'erreur diminue exponentiellement avec la différenciation.
 */
public class ExponentialNoiseModel implements INoiseModel {

    private final double beta; // Paramètre de sensibilité

    public ExponentialNoiseModel(double beta) {
        this.beta = beta;
    }

    @Override
    public double getErrorProbability(double differentiation) {
        // Formule: sigma = 0.5 * exp(-beta * Theta)
        return 0.5 * Math.exp(-beta * differentiation);
    }
}