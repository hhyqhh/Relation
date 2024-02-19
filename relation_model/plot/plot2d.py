

import numpy as np
from problem.utils import get_data
import matplotlib.pyplot as plt





def show_predict_result(prob,model,show_true=False):
    # 判断 prob 是否属于pymoo.core.problem
    from pymoo.core.problem import Problem
    if not isinstance(prob,Problem):
        raise ValueError("prob is not a pymoo.core.problem")

    Xs,ys = get_data(prob,num=100,method='HLS')

    x = np.linspace(prob.xl[0],prob.xu[0],200)
    y = np.linspace(prob.xl[1],prob.xu[1],200)

    X,Y = np.meshgrid(x,y)
    XYs = np.vstack([X.flatten(),Y.flatten()]).T
    Z = prob.evaluate(XYs).reshape(X.shape)

    model.fit(Xs,ys)
    Z_pre = model.predict(XYs).reshape(X.shape)
    if show_true:
        Z_pre_true = model.predict_by_real(XYs,prob.evaluate).reshape(X.shape)
        Z_pre_true = Z_pre_true - Z_pre_true.min()/ (Z_pre_true.max() - Z_pre_true.min())

    # normalize
    Z = Z - Z.min()/ (Z.max() - Z.min())
    Z_pre = Z_pre - Z_pre.min()/ (Z_pre.max() - Z_pre.min())


    if show_true:
        fig,axs = plt.subplots(1,3,figsize=(12,4))
    else:
        fig,axs = plt.subplots(1,2,figsize=(8,4))


    axs[0].contourf(X,Y,Z,cmap='Blues')
    axs[0].set_title('real fitness')
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')

    axs[1].contourf(X,Y,-Z_pre,cmap='Blues')
    axs[1].set_title('predict fitness')
    axs[1].set_xlabel('x1')
    axs[1].set_ylabel('x2')

    if show_true:

        axs[2].contourf(X,Y,-Z_pre_true,cmap='Blues')
        axs[2].set_title('predict fitness by real (only for test)')
        axs[2].set_xlabel('x1')
        axs[2].set_ylabel('x2')


    fig.suptitle(f"{model.__class__.__name__} of {prob.__class__.__name__}")
    plt.tight_layout()

    plt.show()





if __name__ == "__main__":
    from problem.single.LZG import LZG01,LZG02,LZG03,LZG04
    from relation_model.model.single_objective import FitnessCriteria_RSO,CategoryCriteria_RSO
    prob = LZG01(n_var=2)
    # model = FitnessCriteria_RSO()
    # show_predict_result(prob,model,show_true=True)

    model = CategoryCriteria_RSO()
    show_predict_result(prob,model,show_true=True)
