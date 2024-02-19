from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population
import numpy as np
import copy
import random
from algorithm.utils.edamodel import local_search, VWH


def get_best_solution(pop):
    """
    返回函数值最小的个体(solution)
    """
    I = np.argsort(pop.get('F').flatten())
    pop_sorted = copy.deepcopy(pop)
    pop_sorted = pop_sorted[I]
    return pop_sorted[0]


def check_boundary(x, lb, ub):
    """
    边界校验
    """
    # check upper boundary
    ind = x > ub
    x[ind] = np.max(np.vstack((2 * ub[ind] - x[ind], lb[ind])), 0)
    # check lower boundary
    ind = x < lb
    x[ind] = np.min(np.vstack((2 * lb[ind] - x[ind], ub[ind])), 0)

    return x


def get_random_index(current_index, population_size):
    """
    在当前种群索引中获得5个不同于current_index 的随机索引
    """
    res_index = [current_index]
    r = current_index
    for _ in range(5):
        while r in res_index:
            r = random.randint(0, population_size - 1)
        res_index.append(r)
    return res_index[1:]


class CoDE_Reproduction(InfillCriterion):
    def __init__(self, parameter_pool=None,
                 trail_vectors_num=3,
                 **kwargs):
        super().__init__(**kwargs)
        if parameter_pool is None:
            parameter_pool = ((1.0, 0.1), (1.0, 0.9), (0.8, 0.2))
        self.parameter_pool = parameter_pool
        self.trail_vectors_num = trail_vectors_num

    def do(self, problem, pop, n_offsprings, **kwargs):
        lb = problem.xl
        ub = problem.xu

        xs, ys = pop.get("X"), pop.get("F")
        s_best = get_best_solution(pop)

        # ts = np.tile(xs, [self.trail_vectors_num, 1])
        ts = np.zeros(shape=(xs.shape[0] * self.trail_vectors_num, xs.shape[1]))
        for i in range(xs.shape[0]):
            # rand/1/bin;
            ts[i * 3, :] = self.opt1(i, xs, lb, ub)
            # rand/2/bin
            ts[i * 3 + 1, :] = self.opt2(i, xs, lb, ub)
            # current-to-best/1
            ts[i * 3 + 2, :] = self.opt3(i, xs, lb, ub, s_best.get('X'))

            # # rand/1/bin;
            # ts[i, :] = self.opt1(i, xs, lb, ub)
            # # rand/2/bin
            # ts[xs.shape[0] + i, :] = self.opt2(i, xs, lb, ub)
            # # current-to-best/1
            # ts[xs.shape[0] * 2 + i, :] = self.opt3(i, xs, lb, ub, s_best.get('X'))

        trials = Population.new(X=ts)
        return trials

    def opt1(self, x_index, xs, lb, ub):
        # rand/1/bin;
        random_index = get_random_index(x_index, xs.shape[0])
        x = copy.deepcopy(xs[x_index, :])
        F, CR = self.parameter_pool[random.randint(0, len(self.parameter_pool) - 1)]
        jrand = random.randint(0, xs.shape[1] - 1)

        u = xs[random_index[0], :] + F * (xs[random_index[1], :] - xs[random_index[2], :])
        u = check_boundary(u, lb, ub)
        cross_ind = np.random.random(xs.shape[1]) < CR
        cross_ind[jrand] = True
        x[cross_ind] = u[cross_ind]
        return x

    def opt2(self, x_index, xs, lb, ub):
        # rand/2/bin
        random_index = get_random_index(x_index, xs.shape[0])
        x = copy.deepcopy(xs[x_index, :])
        F, CR = self.parameter_pool[random.randint(0, len(self.parameter_pool) - 1)]
        jrand = random.randint(0, xs.shape[1] - 1)

        u = xs[random_index[0]] + F * (xs[random_index[1], :] - xs[random_index[2], :]) + F * (
                xs[random_index[3], :] - xs[random_index[4], :])
        u = check_boundary(u, lb, ub)
        cross_ind = np.random.random(xs.shape[1]) < CR
        cross_ind[jrand] = True
        x[cross_ind] = u[cross_ind]
        return x

    def opt3(self, x_index, xs, lb, ub, x_best):
        random_index = get_random_index(x_index, xs.shape[0])
        # x = copy.deepcopy(xs[x_index, :])
        F = self.parameter_pool[random.randint(0, len(self.parameter_pool) - 1)][0]

        u = xs[x_index, :] + F * (x_best - xs[x_index, :]) + F * (xs[random_index[0], :] - xs[random_index[1], :])
        u = check_boundary(u, lb, ub)
        return u


class VWH_Local_Reproduction(InfillCriterion):
    def __init__(self, eda=VWH(M=15), Pb=0.2, Pc=0.2, **kwargs):
        super().__init__(**kwargs)
        self.eda = eda
        self.Pb = Pb
        self.Pc = Pc

    def do(self, problem, pop, n_offsprings, **kwargs):
        # 更新 eda 模型
        algorithm = kwargs['algorithm']
        xs, ys = pop.get('X'), pop.get('F')
        I = np.argsort(ys.flatten())
        xs = xs[I, :]
        ys = ys[I]
        self.eda.update(xs)

        # 采样 新解
        xs_eda = self.eda.sample(n_offsprings)

        # local search 提升解质量
        NL = int(np.floor(algorithm.pop_size * self.Pb))
        xs_ls = local_search(xs[:NL, :], ys[:NL])

        # Crossover
        I = np.floor(np.random.random((algorithm.pop_size, 1)) * (xs_ls.shape[0] - 2)).astype(int).flatten()
        xtmp = xs_ls[I, :]
        mask = np.random.random((algorithm.pop_size, problem.n_var)) < self.Pc
        xs_eda[mask] = xtmp[mask]

        # boundary checking
        lb_matrix = problem.xl * np.ones(shape=xs_eda.shape)
        ub_matrix = problem.xu * np.ones(shape=xs_eda.shape)

        pos = xs_eda < problem.xl
        xs_eda[pos] = 0.5 * (xs[pos] + lb_matrix[pos])

        pos = xs_eda > problem.xu
        xs_eda[pos] = 0.5 * (xs[pos] + ub_matrix[pos])

        trials = Population.new(X=xs_eda)
        return trials


class VWH_Local_Reproduction_unevaluate(InfillCriterion):
    def __init__(self, eda=VWH(M=15), Pb=0.2, Pc=0.2, **kwargs):
        super().__init__(**kwargs)
        self.eda = eda
        self.Pb = Pb
        self.Pc = Pc

    def do(self, problem, pop, n_offsprings, **kwargs):
        # 更新 eda 模型
        algorithm = kwargs['algorithm']
        unevaluated_pop = kwargs['unevaluated_pop']
        xs, ys = pop.get('X'), pop.get('F')
        I = np.argsort(ys.flatten())
        xs = xs[I, :]
        ys = ys[I]

        self.eda.update(np.concatenate([xs, unevaluated_pop[:int(algorithm.pop_size / 2), :]], axis=0))
        # self.eda.update(xs)

        # 采样 新解
        xs_eda = self.eda.sample(n_offsprings)

        # local search 提升解质量
        NL = int(np.floor(algorithm.pop_size * self.Pb))
        xs_ls = local_search(xs[:NL, :], ys[:NL])

        # Crossover
        I = np.floor(np.random.random((algorithm.pop_size, 1)) * (xs_ls.shape[0] - 2)).astype(int).flatten()
        xtmp = xs_ls[I, :]
        mask = np.random.random((algorithm.pop_size, problem.n_var)) < self.Pc
        xs_eda[mask] = xtmp[mask]

        # boundary checking
        lb_matrix = problem.xl * np.ones(shape=xs_eda.shape)
        ub_matrix = problem.xu * np.ones(shape=xs_eda.shape)

        pos = xs_eda < problem.xl
        xs_eda[pos] = 0.5 * (xs[pos] + lb_matrix[pos])

        pos = xs_eda > problem.xu
        xs_eda[pos] = 0.5 * (xs[pos] + ub_matrix[pos])

        trials = Population.new(X=xs_eda)
        return trials



