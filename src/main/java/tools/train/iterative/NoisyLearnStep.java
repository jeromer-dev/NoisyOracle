package tools.train.iterative;

import java.util.List;
import lombok.Getter;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.ranking.Ranking;
import tools.train.LearnStep;

/**
 * Implémentation concrète de l'interface LearnStep pour transmettre
 * l'état de l'apprentissage aux observateurs (ex: ExperimentLogger).
 */
public class NoisyLearnStep implements LearnStep {

    @Getter
    private final ISinglevariateFunction currentScoreFunction;
    
    @Getter
    private final List<Ranking<IAlternative>> rankingHistory;
    
    @Getter
    private final int iteration;

    public NoisyLearnStep(ISinglevariateFunction currentScoreFunction, List<Ranking<IAlternative>> rankingHistory, int iteration) {
        this.currentScoreFunction = currentScoreFunction;
        this.rankingHistory = rankingHistory;
        this.iteration = iteration;
    }
}