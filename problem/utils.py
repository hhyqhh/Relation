from pyDOE2 import lhs
from pymoo.core.problem import Problem
import numpy as np




def get_data(prob,num=100,method='HLS'):
    # 判断是否是pymoo的problem
    if not isinstance(prob, Problem):
        raise ValueError("prob must be a pymoo Problem")
    # get the data
    lb,ub = prob.xl,prob.xu
    if method == 'HLS':
        Xs = lhs(prob.n_var, num, criterion='center')
        Xs = lb + (ub - lb) * Xs
    elif method == 'uniform':
        Xs = np.random.uniform(low=lb,high=ub,size=(num,prob.n_var))
    elif method == 'random':
        Xs = np.random.random(size=(num,prob.n_var))    
        Xs = lb + (ub - lb) * Xs
    else :
        raise ValueError("method must be HLS or uniform or random")
    ys = prob.evaluate(Xs)
    return Xs,ys