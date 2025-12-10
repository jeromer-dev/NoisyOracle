package tools.utils;

/**
 * Gère les paramètres des modèles de bruit et le seuil de sécurité adaptatif.
 */
public class NoiseModelConfig {

    private final int totalBudgetL;
    private final double tauInit = 0.5; 
    private final double gamma;           
    private final double linearNoiseK; 
    private final double expNoiseBeta; 
    
    public NoiseModelConfig(int totalBudgetL, double linearNoiseK, double expNoiseBeta) {
        this.totalBudgetL = totalBudgetL;
        this.gamma = totalBudgetL > 0 ? 4.0 / totalBudgetL : 0.0;
        this.linearNoiseK = linearNoiseK;
        this.expNoiseBeta = expNoiseBeta;
    }

    public double getAdaptiveThreshold(int t) {
        if (t < 0) return tauInit;
        if (t > totalBudgetL) t = totalBudgetL; 
        
        return tauInit * Math.exp(-gamma * t);
    }
    
    public int getTotalBudgetL() {
        return totalBudgetL;
    }

    public double getLinearNoiseK() {
        return linearNoiseK;
    }

    public double getExpNoiseBeta() {
        return expNoiseBeta;
    }
}