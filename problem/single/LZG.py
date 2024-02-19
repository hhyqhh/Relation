from pymoo.core.problem import Problem
import numpy as np
from math import pi, cos, sin, exp, sqrt, e


class LZG01(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, xl=-5.12, xu=+5.12)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            for j in range(d):
                ys[i] += (j + 1) * xi[j] * xi[j]
        out["F"] = ys


class LZG02(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, xl=-2.048 , xu=+2.048)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            for j in range(d - 1):
                ys[i] += (100 * (xi[j + 1] - xi[j] * xi[j]) ** 2 + (1 - xi[j]) ** 2)
        out["F"] = ys


class LZG03(Problem):
    def __init__(self, n_var=2):
        super().__init__(n_var=n_var, n_obj=1, xl=-32.768, xu=+32.768)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            re1, re2 = 0, 0
            for j in range(d):
                re1 += xi[j] * xi[j]
                re2 += cos(2 * pi * xi[j])
            re1 = re1 / d
            re2 = re2 / d
            ys[i] = -20 * exp(-0.2 * sqrt(re1)) - exp(re2) + 20 + e
        out["F"] = ys


class LZG04(Problem):
    def __init__(self, n_var=2,lx=-600,ux=600):
        super().__init__(n_var=n_var, n_obj=1, xl=lx, xu=ux)

    def _evaluate(self, x, out, *args, **kwargs):
        n, d = x.shape
        ys = np.zeros(shape=(n,))
        for i, xi in enumerate(x):
            re1, re2, result = 0, 1, 0
            for j in range(d):
                re1 += xi[j] * xi[j] / 4000
                re2 *= cos(xi[j] / sqrt(j + 1))
            result = 1 + re1 - re2
            ys[i] = result
        out["F"] = ys





