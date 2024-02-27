from pymoo.optimize import minimize
from problem.single.LZG import LZG01, LZG02, LZG03, LZG04
from loguru import logger
from algorithm.saea.drso import DRSO



logger.remove()
problem = LZG01(n_var=20)
algorithm = DRSO()
res = minimize(problem,
                algorithm,
                ('n_evals', 500),
                verbose=True)

