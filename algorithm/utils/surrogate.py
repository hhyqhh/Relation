from sklearn.ensemble import RandomForestRegressor as _sk_RandomForestRegressor
import numpy as np
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def get_surrogate(surrogate_type):
    # check the surrogate model
    if surrogate_type not in ["XGB","RF","GP","Relation"]:
        raise Exception("surrogate_type is not supported")
    if surrogate_type == "XGB":
        surrogate =  xgb.XGBRegressor(eval_metric='logloss')
    elif surrogate_type == "RF":
        surrogate =  RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
    elif surrogate_type == "GP":
        surrogate =  GaussianProcessRegressor()
    else: 
        raise Exception("surrogate_type is not supported")
    return surrogate


def GP_wrapper(surrogate,n_dims=10):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    # 忽略ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


    cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
    other_kernel = Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)
    surrogate.set_params(normalize_y=True,n_restarts_optimizer=2,kernel=cov_amplitude * other_kernel)
    return surrogate






class RandomForestRegressor(_sk_RandomForestRegressor):
    """
    RandomForestRegressor that supports conditional std computation.
    
    Code reference from skopt/learning/forest.py 

    Fix the bug caused by the update of sklearn version: 
        sklearn.utils._param_validation.InvalidParameterError: 
        The 'criterion' parameter of RandomForestRegressor must be a str 
        among {'poisson', 'friedman_mse', 'squared_error', 'absolute_error'}. Got 'mse' instead.

        Change    max_features='auto' to    max_features='sqrt'
        Change    criterion='mse'     to    criterion='squared_error' 
    """
    
    def __init__(self, n_estimators=10, criterion='squared_error', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='sqrt',
                 max_leaf_nodes=None, min_impurity_decrease=0.,
                 bootstrap=True, oob_score=False,
                 n_jobs=1, random_state=None, verbose=0, warm_start=False,
                 min_variance=0.0):
        self.min_variance = min_variance
        super(RandomForestRegressor, self).__init__(
            n_estimators=n_estimators, criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap, oob_score=oob_score,
            n_jobs=n_jobs, random_state=random_state,
            verbose=verbose, warm_start=warm_start)
        
    def predict(self, X, return_std=False):
        """Predict continuous output for X.

        Parameters
        ----------
        X : array of shape = (n_samples, n_features)
            Input data.

        return_std : boolean
            Whether or not to return the standard deviation.

        Returns
        -------
        predictions : array-like of shape = (n_samples,)
            Predicted values for X. If criterion is set to "squared_error",
            then `predictions[i] ~= mean(y | X[i])`.

        std : array-like of shape=(n_samples,)
            Standard deviation of `y` at `X`. If criterion
            is set to "mse", then `std[i] ~= std(y | X[i])`.

        """
        mean = super(RandomForestRegressor, self).predict(X)

        if return_std:
            if self.criterion != "squared_error":
                raise ValueError(
                    "Expected impurity to be 'squared_error', got %s instead"
                    % self.criterion)
            std = self._return_std(X, self.estimators_, mean, self.min_variance)
            return mean, std
        return mean


    def _return_std(self,X, trees, predictions, min_variance):
        """
        Returns `std(Y | X)`.

        Can be calculated by E[Var(Y | Tree)] + Var(E[Y | Tree]) where
        P(Tree) is `1 / len(trees)`.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Input data.

        trees : list, shape=(n_estimators,)
            List of fit sklearn trees as obtained from the ``estimators_``
            attribute of a fit RandomForestRegressor or ExtraTreesRegressor.

        predictions : array-like, shape=(n_samples,)
            Prediction of each data point as returned by RandomForestRegressor
            or ExtraTreesRegressor.

        Returns
        -------
        std : array-like, shape=(n_samples,)
            Standard deviation of `y` at `X`. If criterion
            is set to "mse", then `std[i] ~= std(y | X[i])`.

        """
        # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906
        std = np.zeros(len(X))

        for tree in trees:
            var_tree = tree.tree_.impurity[tree.apply(X)]

            # This rounding off is done in accordance with the
            # adjustment done in section 4.3.3
            # of http://arxiv.org/pdf/1211.0906v2.pdf to account
            # for cases such as leaves with 1 sample in which there
            # is zero variance.
            var_tree[var_tree < min_variance] = min_variance
            mean_tree = tree.predict(X)
            std += var_tree + mean_tree ** 2

        std /= len(trees)
        std -= predictions ** 2.0
        std[std < 0.0] = 0.0
        std = std ** 0.5
        return std




