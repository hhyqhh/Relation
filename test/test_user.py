from relation_model.component.generator import FitnessCriteriaGenerator,CategoryCriteriaGenerator
from relation_model.component.user import FitnessCriteriaUser,CategoryCriteriaUser
from relation_model.component.trainer import Trainer
import pytest
from problem.single.LZG import LZG01,LZG02,LZG03,LZG04
from problem.utils import get_data
import numpy as np



def test_FitnessCriteria_pipeline():
    """
    测试FitnessCriteria的流程,没有校对精度
    """

    problem = LZG03(n_var=20)
    Xs,ys = get_data(problem,num=100,method='HLS')

    fg = FitnessCriteriaGenerator()
    res = fg.do(Xs,ys)
    RXs = res['RXs']
    Rls = res['Rls']

    trainer = Trainer()
    trainer.do(RXs,Rls)

    Us,tys = get_data(problem,50)
    user= FitnessCriteriaUser(generator=fg,trainer=trainer)

    pred_ys,sort_ind=user.do(Us,return_index=True)
    print(pred_ys)
    print(sort_ind)
    print('------------------')

    real_ys,real_sort_ind = user.real_do(Us,problem.evaluate,return_index=True)
    print(real_ys)
    print(real_sort_ind)
    print('------------------')
    t_sort_ind =  tys.flatten().argsort()     
    print(t_sort_ind)
    # assert (real_sort_ind == t_sort_ind).all()


def test_CategoryCriteria_pipeline():
    """
    测试CategoryCriteria的流程,没有校对精度
    """
    problem = LZG01(n_var=50)
    Xs,ys = get_data(problem,num=100,method='HLS')
    cg = CategoryCriteriaGenerator()
    res = cg.do(Xs,ys)
    RXs = res['RXs']
    Rls = res['Rls']

    trainer = Trainer()
    trainer.do(RXs,Rls)

    Us,tys = get_data(problem,50)
    user = CategoryCriteriaUser(generator=cg,trainer=trainer)

    pred_ys,_=user.do(Us,return_index=True)
    print(pred_ys)
    print('------------------')

    real_pred_ys = user.real_do(Us,problem.evaluate)
    print(real_pred_ys)
    print('------------------')

    split_boundary = cg.get_split_result()['split_boundary']
    t_ys = np.array([1 if y < split_boundary else -1 for y in tys.flatten()])
    print(t_ys)









def test_User_check_compatibility():
    generator1 = FitnessCriteriaGenerator()
    generator2 = CategoryCriteriaGenerator()

    user = FitnessCriteriaUser(generator=generator1,trainer=Trainer())
    user = CategoryCriteriaUser(generator=generator2,trainer=Trainer())

    with pytest.raises(ValueError):
        user = FitnessCriteriaUser(generator=generator2,trainer=Trainer())  

    with pytest.raises(ValueError):
        user = CategoryCriteriaUser(generator=generator1,trainer=Trainer())

if __name__ == "__main__":
    # test_User_check_compatibility()

    # test_FitnessCriteria_pipeline()
    # test_CategoryCriteria_pipeline()