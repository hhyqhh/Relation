from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.replacement import ImprovementReplacement
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.optimize import minimize
from problem.single.LZG import LZG01, LZG02, LZG03, LZG04
from algorithm.utils.reproduction import CoDE_Reproduction
import numpy as np
from pymoo.util.display.single import SingleObjectiveOutput

class CompositeDE(GeneticAlgorithm):
    def __init__(self,
                 pop_size=30,
                 n_offsprings=None,
                 sampling=FloatRandomSampling(),
                 reproduction=CoDE_Reproduction(),
                 **kwargs
                 ):
        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         **kwargs)

        self.reproduction = reproduction

    def _infill(self):
        infills = self.reproduction.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        index = np.arange(len(infills))
        infills.set("index", index)
        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        infills = self.preselection_by_real(infills)
        I = range(self.n_offsprings)
        infills.set('index', I)

        self.pop[I] = ImprovementReplacement().do(self.problem, self.pop[I], infills)

        FitnessSurvival().do(self.problem, self.pop, return_indices=True)

    def preselection_by_real(self, trials):
        ys = trials.get('F')
        index = trials.get('index')

        ys_ = ys.flatten().reshape(-1, self.reproduction.trail_vectors_num)
        index_ = np.array(index).reshape(-1, self.reproduction.trail_vectors_num)
        I_selected = np.argmin(ys_, axis=1)
        I = np.full(fill_value=False, shape=trials.shape)
        for i in range(self.n_offsprings):
            I[index_[i, I_selected[i]]] = True

        return trials[I]



if __name__ == '__main__':

    problem = LZG01(n_var=20)
    algorithm = CompositeDE(pop_size=50)
    from pymoo.optimize import minimize
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 1000),
                   seed=1,
                   verbose=True,output=SingleObjectiveOutput(),)
    print("hash", res.F.sum())