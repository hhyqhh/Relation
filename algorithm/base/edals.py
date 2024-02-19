from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from algorithm.utils.reproduction import VWH_Local_Reproduction
import numpy as np
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from scipy.optimize import minimize as scipy_minimize
import random
from pymoo.util.display.single import SingleObjectiveOutput

class f1(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, xl=-100, xu=+100)

    def _evaluate(self, x, out, *args, **kwargs):
        value = np.sum(np.square(x), axis=1)
        out["F"] = value


class ObjectiveWrapper(object):
    def __init__(self, prob):
        if prob is None:
            raise Exception('prob is None')
        self.prob = prob
        self.eva_count = 0
        self.Xs = []
        self.ys = []

    def evaluate(self, x):
        self.eva_count += 1
        y = self.prob.evaluate(x)
        self.Xs.append(x)
        self.ys.append(y)
        return y

    def get(self, opt):
        if opt == 'Xs':
            return np.array(self.Xs)
        elif opt == 'ys':
            return np.array(self.ys).reshape(-1, 1)
        else:
            raise Exception('opt is not Xs or ys')

    def get_best(self):
        best_index = np.argmin(self.ys)
        return self.Xs[best_index], self.ys[best_index]

    def get_eva_count(self):
        return self.eva_count


class EDALS(GeneticAlgorithm):
    """
    EDALS
    2023.7.30
    加入Powell Search 方法， 目前支持mini 方法。
    """

    def __init__(self,
                 pop_size=30,
                 n_offsprings=None,
                 sampling=FloatRandomSampling(),
                 output=SingleObjectiveOutput(),
                 reproduction=VWH_Local_Reproduction(),
                 ps_method=False,
                 theta=0.1,
                 **kwargs
                 ):
        super().__init__(pop_size=pop_size,
                         n_offsprings=n_offsprings,
                         sampling=sampling,
                         output=output,
                         survival=FitnessSurvival(),
                         **kwargs)

        self.ps_method = ps_method
        self.reproduction = reproduction

        self.nls = 0
        self.best_current_fitness = []
        self.theta = theta

    def _initialize_advance(self, infills=None, **kwargs):
        # init the eda model
        self.reproduction.eda.init(
            D=self.problem.n_var,
            LB=self.problem.xl * np.ones(shape=self.problem.n_var),
            UB=self.problem.xu * np.ones(shape=self.problem.n_var)
        )

        # save current best fitness
        self.best_current_fitness.append(min(self.pop.get('F')))

    def _infill(self):
        # 利用 VWH 模型采样产生新解
        infills = self.reproduction.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        index = np.arange(len(infills))
        infills.set('index', index)
        return infills

    def _advance(self, infills=None, **kwargs):
        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # execute the survival to find the fittest solutions
        pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)
        self.best_current_fitness.append(min(pop.get('F')))
        best_current_fitness = np.array(self.best_current_fitness)
        """
        需要加入 ps 方法
        """
        if self.ps_method and (self.n_gen - self.nls) > 50:
            xrange = np.mean(np.array(self.reproduction.eda.UB_list) - np.array(self.reproduction.eda.LB_list),
                             axis=1).flatten()
            cx = abs(xrange[self.n_gen - 50 - 1] - xrange[self.n_gen - 1]) / (
                    max(xrange[self.n_gen - 1], xrange[self.n_gen - 50 - 1]) + np.finfo(
                np.float32).eps)  # condition in X - space
            cf = abs(best_current_fitness[self.n_gen - 50 - 1] - best_current_fitness[self.n_gen - 1]) / (
                    best_current_fitness[self.n_gen - 50 - 1] + np.finfo(np.float32).eps)
            if min(cx, cf) < self.theta:
                # random select one solution from pop
                # t_index = random.randint(0, self.pop_size - 1)
                # x0 = pop.get('X')[t_index]

                # select the best solution
                t_ys = pop.get('F')
                t_xs = pop.get('X')
                t_index = np.argmin(t_ys)
                x0 = t_xs[t_index, :]

                bounds = [(l, u) for l, u in zip(self.problem.xl, self.problem.xu)]
                objw = ObjectiveWrapper(self.problem)

                if hasattr(self.termination, 'n_max_gen'):
                    maxiter = np.floor((self.termination.n_max_gen - self.n_gen) * 0.5)
                else:
                    maxiter = np.floor((self.termination.n_max_evals / self.pop_size - self.n_gen) * 0.5)

                _ = scipy_minimize(objw.evaluate, x0, bounds=bounds, method='Powell', options={'maxiter': maxiter})

                x_best, y_best = objw.get_best()
                pop[t_index].set('X', x_best)
                pop[t_index].set('F', y_best)
                # x_best = Population.new('X', x_best.reshape(1, -1))
                # x_best.set('F', y_best)
                #
                # pop[random_index] = x_best[0]

                self.evaluator.n_eval += objw.get_eva_count()
                self.nls = self.n_gen

        self.pop = pop


if __name__ == '__main__':
    from pymoo.optimize import minimize
    from problem.single.LZG import LZG01, LZG02, LZG03, LZG04

    # Prob = create_problem_class('F12005', 20)
    # problem = Prob()

    # problem = f1(n_var=20)
    problem = LZG01(n_var=20)
    algorithm = EDALS(pop_size=50, ps_method=True)


    res = minimize(problem,
                   algorithm,
                   ('n_evals', 10000),
                   # ('n_gen', 1000),
                   verbose=True)
    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))
