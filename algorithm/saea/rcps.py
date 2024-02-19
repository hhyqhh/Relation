from pymoo.operators.sampling.rnd import FloatRandomSampling
from algorithm.base.composite_de import CompositeDE
from algorithm.utils.reproduction import CoDE_Reproduction
from problem.single.LZG import LZG01, LZG02, LZG03, LZG04
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.core.replacement import ImprovementReplacement
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from relation_model.model.single_objective import RelationPreselection
import numpy as np

class RCPS_CoDE(CompositeDE):
    def __init__(self, pop_size=30, n_offsprings=None, **kwargs):
        super().__init__(pop_size, n_offsprings, **kwargs)

        self.model = RelationPreselection()



    def next(self):

        # get the infill solutions
        infills = self.infill()

        # call the advance with them after evaluation
        if infills is not None:
            if self.n_gen > 1:
                infills = self.preselection_by_relation(infills)

            self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)

        # if the algorithm does not follow the infill-advance scheme just call advance
        else:
            self.advance()

    def _infill(self):

        Xs,ys = self.pop.get("X"),self.pop.get("F")
        self.model.fit(Xs,ys)
        

        infills = self.reproduction.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        index = np.arange(len(infills))
        infills.set("index", index)
        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        I = range(self.n_offsprings)
        infills.set('index', I)

        self.pop[I] = ImprovementReplacement().do(self.problem, self.pop[I], infills)

        FitnessSurvival().do(self.problem, self.pop, return_indices=True)







    def preselection_by_relation(self, trials):
        trail_vectors_num = self.reproduction.trail_vectors_num
        index = trials.get('index')

        trials_ = [trials[i:i+trail_vectors_num] for i in range(0, len(trials), trail_vectors_num)]

        

        scores = np.array([self.model.predict(trial.get("X")) for trial in trials_]).reshape(-1, trail_vectors_num)

        index_ = np.array(index).reshape(-1, self.reproduction.trail_vectors_num)

        I_selected = np.argmax(scores, axis=1)
        I = np.full(fill_value=False, shape=trials.shape)
        for i in range(self.n_offsprings):
            I[index_[i, I_selected[i]]] = True

        return trials[I]



        




if __name__ == '__main__':

    problem = LZG02(n_var=10)
    algorithm = RCPS_CoDE(pop_size=50)
    from pymoo.optimize import minimize
    res = minimize(problem,
                   algorithm,
                   ('n_evals', 15000),
                   seed=1,
                   verbose=True,output=SingleObjectiveOutput(),)
    print("hash", res.F.sum())