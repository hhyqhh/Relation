from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from algorithm.utils.reproduction import VWH_Local_Reproduction_unevaluate
from relation_model.model.single_objective import FitnessCriteria_RSO, CategoryCriteria_RSO
import copy
import numpy as np
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival




class DRSO(GeneticAlgorithm):
    def __init__(self,
                 pop_size=50,
                 tao=100,
                 sample=LHS(),
                 reproduction=VWH_Local_Reproduction_unevaluate(),
                 output=SingleObjectiveOutput(),
                 c1_model = FitnessCriteria_RSO(),
                 c2_model = CategoryCriteria_RSO(),
                 display = SingleObjectiveOutput(),
                 **kwargs):
        super().__init__(pop_size=pop_size,
                         sampling=sample,
                         output=output,
                         survival = FitnessSurvival(),
                         **kwargs)
        
        self.reproduction = reproduction
        self.tao = tao
        self.M1 = c1_model
        self.M2 = c2_model

        self.unevaluated_pop = None
        self.archive_eva = Population()


    def _initialize_advance(self, infills=None, **kwargs):
        self.reproduction.eda.init(
            D=self.problem.n_var,
            LB=self.problem.xl * np.ones(shape=self.problem.n_var),
            UB=self.problem.xu * np.ones(shape=self.problem.n_var)
        )
        # 将初始化种群保存至archive_eva
        self.archive_eva = Population.merge(self.archive_eva, self.pop)

        self.unevaluated_pop = copy.deepcopy(self.pop.get('X'))


    def get_raw_training_data(self):
        """
        从 archive_eva 中选择tao 个解返回
        """
        t_xs, t_ys = self.archive_eva.get("X"), self.archive_eva.get("F")
        if len(self.archive_eva) <= self.tao:
            return t_xs, t_ys.flatten()
        else:
            t = copy.deepcopy(t_ys).flatten()
            index = t.argsort()
            return t_xs[index[: self.tao], :], t_ys[index[: self.tao], :].flatten()



    def _infill(self):
        # get current population
        t_xs, t_ys = self.get_raw_training_data()
        # train surrogate model
        self.training_surrogete_model(t_xs, t_ys)
        # get the infill solutions
        infills = self.reproduction.do(
            self.problem,
            self.pop,
            self.n_offsprings,
            algorithm=self,
            unevaluated_pop=self.unevaluated_pop
        )
        # surrogete assisted selection
        x_best, unevaluated_pop = self.surrogate_assisted_selection(infills)
        self.unevaluated_pop = unevaluated_pop

        infills = Population.new(X=x_best)
        return infills
    

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self.archive_eva = Population.merge(self.archive_eva, infills)

        self.pop = self.survival.do(
            self.problem, self.archive_eva, n_survive=self.pop_size, algorithm=self, **kwargs)


    def training_surrogete_model(self, Xs, ys):
        self.M1.fit(Xs, ys)
        self.M2.fit(Xs, ys)

    def surrogate_assisted_selection(self, pop):
        Xs = pop.get('X')
        ys_pre1 = self.M1.predict(Xs)
        ys_pre2 = self.M2.predict(Xs)

        # 选择最优的解
        sorted_ind = np.argsort(ys_pre1.flatten())
        X_best = copy.deepcopy(Xs[sorted_ind[-1], :]).reshape(1, -1)

        # 选择unevaluated_pop
        selected = ys_pre2.flatten() > 0
        selected_decs = copy.deepcopy(Xs)[selected, :]

        if selected_decs.shape[0] > 25:
            r_index = np.random.permutation(selected_decs.shape[0])
            selected_decs = selected_decs[r_index[:int(self.pop_size / 2)], :]

        return X_best, selected_decs
    


if __name__ == '__main__':
    from pymoo.optimize import minimize
    from problem.single.LZG import LZG01, LZG02, LZG03, LZG04
    from loguru import logger

    logger.remove()

    problem = LZG02(n_var=20)
    algorithm = DRSO()
    res = minimize(problem,
                   algorithm,
                   ('n_evals', 500),
                   verbose=True)