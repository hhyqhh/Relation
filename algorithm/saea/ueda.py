from typing import Any
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from algorithm.utils.reproduction import VWH_Local_Reproduction_unevaluate
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.single import SingleObjectiveOutput
import copy
import numpy as np
from skopt.utils import cook_estimator
import xgboost as xgb
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor


from algorithm.utils.surrogate import RandomForestRegressor, get_surrogate,GP_wrapper



"""

@misc{hao2024model,
      title={Model Uncertainty in Evolutionary Optimization and Bayesian Optimization: A Comparative Analysis}, 
      author={Hao Hao and Xiaoqun Zhang and Aimin Zhou},
      year={2024},
      eprint={2403.14413},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
"""





class UEDA(GeneticAlgorithm):
    def __init__(self,
                 pop_size=50,
                 tao=100,  # yo
                 sampling=LHS(),
                 reproduction=VWH_Local_Reproduction_unevaluate(),
                 output=SingleObjectiveOutput(),
                 surrogate=None,
                 **kwargs):
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         output=output,
                         survival=FitnessSurvival(),
                         **kwargs)
        self.reproduction = reproduction

        # all solutions that have been evaluated so far
        self.archive_eva = Population()
        self.current_sored_Xs = None
        self.tao = tao

        self.surrogate = surrogate



    def _initialize_advance(self, infills=None, **kwargs):
        # init the eda model
        self.reproduction.eda.init(
            D=self.problem.n_var,
            LB=self.problem.xl * np.ones(shape=self.problem.n_var),
            UB=self.problem.xu * np.ones(shape=self.problem.n_var)
        )

        # 将初始化种群保存至Archive
        self.archive_eva = Population.merge(self.pop, self.archive_eva)

        self.unevaluated_pop = copy.deepcopy(self.pop.get('X'))

        if self.surrogate is None:
            raise Exception("surrogate model is None")



    def _infill(self):
        # get current population
        t_xs, t_ys = self.get_raw_training_data()

        # train surrogate model
        self.training_surrogete_model(t_xs, t_ys)

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

    def surrogate_assisted_selection(self, pop):
        Xs = pop.get('X')
        ys_pre = self.surrogate.predict(Xs)

        # 选择最优的解
        sorted_ind = np.argsort(ys_pre.flatten())
        X_best = copy.deepcopy(Xs[sorted_ind[0], :]).reshape(1, -1)

        # 选择unevaluated_pop
        selected_decs = copy.deepcopy(
            Xs)[sorted_ind[:int(self.pop_size / 2)], :]

        # selected = ys_pre.flatten() > 0
        # selected_decs = copy.deepcopy(Xs)[selected, :]
        #
        # if selected_decs.shape[0] > 25:
        #     r_index = np.random.permutation(selected_decs.shape[0])
        #     selected_decs = selected_decs[r_index[:int(self.pop_size / 2)], :]

        return X_best, selected_decs

    def get_raw_training_data(self):
        """
        从 archive 中选择tao 个解返回
        """
        t_xs, t_ys = self.archive_eva.get("X"), self.archive_eva.get("F")
        if len(self.archive_eva) <= self.tao:
            return t_xs, t_ys.flatten()
        else:
            t = copy.deepcopy(t_ys).flatten()
            index = t.argsort()
            return t_xs[index[: self.tao], :], t_ys[index[: self.tao], :].flatten()

    def training_surrogete_model(self, Xs, ys):
        # print("training_surrogete_model, Xs shape:", Xs.shape)
        self.surrogate.fit(Xs, ys)


class UEDA_RF(UEDA):
    def __init__(self, 
                 pop_size=50, 
                 tao=100, 
                 sampling=LHS(), 
                 output=SingleObjectiveOutput(),
                 reproduction=VWH_Local_Reproduction_unevaluate(), 
                 **kwargs):
        
        # INIT the surrogate model
        surrogate = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)

        super().__init__(pop_size=pop_size, 
                         tao=tao, 
                         sampling=sampling, 
                         output=output, 
                         reproduction=reproduction, 
                         surrogate=surrogate, 
                         **kwargs)


class UEDA_XGB(UEDA):
    def __init__(self, 
                pop_size=50, 
                tao=100, 
                sampling=LHS(), 
                output=SingleObjectiveOutput(),
                reproduction=VWH_Local_Reproduction_unevaluate(), 
                **kwargs):
    
        # INIT the surrogate model
        surrogate = xgb.XGBRegressor(eval_metric='logloss')

        super().__init__(pop_size=pop_size, 
                            tao=tao, 
                            sampling=sampling, 
                            output=output, 
                            reproduction=reproduction, 
                            surrogate=surrogate, 
                            **kwargs)
        

class UEDA_GP(UEDA):
    def __init__(self, 
                pop_size=50, 
                tao=100, 
                sampling=LHS(), 
                output=SingleObjectiveOutput(),
                reproduction=VWH_Local_Reproduction_unevaluate(), 
                **kwargs):
    

        super().__init__(pop_size=pop_size, 
                            tao=tao, 
                            sampling=sampling, 
                            output=output, 
                            reproduction=reproduction, 
                            **kwargs)
        

    def _initialize_advance(self, infills=None, **kwargs):
        # init the eda model
        self.reproduction.eda.init(
            D=self.problem.n_var,
            LB=self.problem.xl * np.ones(shape=self.problem.n_var),
            UB=self.problem.xu * np.ones(shape=self.problem.n_var)
        )

        # 将初始化种群保存至Archive
        self.archive_eva = Population.merge(self.pop, self.archive_eva)

        self.unevaluated_pop = copy.deepcopy(self.pop.get('X'))

        # INIT the surrogate model

        self.surrogate = get_surrogate("GP")
        self.surrogate = GP_wrapper(self.surrogate,n_dims=self.problem.n_var)



if __name__=='__main__':
    from problem.single.LZG import LZG01, LZG02, LZG03, LZG04
    from pymoo.optimize import minimize

    problem = LZG04(n_var=20)

    algorithm = UEDA_GP(pop_size=30)
    # algorithm = UEDA_RF(pop_size=30)


    res = minimize(problem,
                   algorithm,
                   ('n_evals', 500),
                   verbose=True)
    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))
    # print(res.algorithm.callback.data["objs"])