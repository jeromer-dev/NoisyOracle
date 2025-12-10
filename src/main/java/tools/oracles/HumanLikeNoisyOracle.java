package tools.oracles;

import tools.alternatives.IAlternative;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule; 
import tools.utils.RandomUtil;

public class HumanLikeNoisyOracle extends ChiSquaredOracle {

    private final INoiseModel noiseModel;
    private final PairwiseUncertainty differentiationFunction;
    private final RandomUtil randomUtil;

    public HumanLikeNoisyOracle(int nbTransactions, INoiseModel noiseModel, PairwiseUncertainty differentiationFunction) {
        super(nbTransactions); 
        this.noiseModel = noiseModel;
        this.randomUtil = new RandomUtil();
        this.differentiationFunction = differentiationFunction;
    }

    /**
     * Méthode publique pour obtenir la préférence bruitée.
     */
    public IAlternative getNoisyPreferredAlternative(IAlternative R1, IAlternative R2, ISinglevariateFunction model) {
        
        IAlternative truePreferred = null;

        // Utilisation de la méthode compare de la classe parente
        try {
            // On cast en DecisionRule car c'est ce que l'oracle attend généralement
            if (R1 instanceof DecisionRule && R2 instanceof DecisionRule) {
                int comparisonResult = super.compare((DecisionRule)R1, (DecisionRule)R2);
                if (comparisonResult > 0) { 
                    truePreferred = R1;
                } else if (comparisonResult < 0) { 
                    truePreferred = R2;
                }
            }
        } catch (Exception e) {
            return null; 
        }

        if (truePreferred == null) {
            return null; 
        }

        double scoreR1 = model.computeScore(R1);
        double scoreR2 = model.computeScore(R2);
        
        double differentiation = differentiationFunction.computeScore(scoreR1, scoreR2); 

        double errorProbability = noiseModel.getErrorProbability(differentiation);

        // Tirage aléatoire pour l'erreur
        boolean errorOccurs = randomUtil.nextDouble() < errorProbability;

        if (errorOccurs) {
            // Retourne l'opposé de la vraie préférence
            if (truePreferred.equals(R1)) {
                return R2;
            } else {
                return R1;
            }
        } else {
            return truePreferred;
        }
    }
}