from relation_model.model.single_objective import FitnessCriteria_RSO,CategoryCriteria_RSO
from problem.single.LZG import LZG01,LZG02,LZG03,LZG04
from relation_model.plot.plot2d import show_predict_result



if __name__ == "__main__":
    """
    测试 2d-plot
    """
    prob = LZG04(n_var=2)
    model = FitnessCriteria_RSO()
    show_predict_result(prob,model,show_true=False)

    model = CategoryCriteria_RSO()
    show_predict_result(prob,model,show_true=False)