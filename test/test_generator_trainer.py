from relation_model.component.generator import FitnessCriteriaGenerator,CategoryCriteriaGenerator
from relation_model.component.trainer import Trainer
from problem.single.LZG import LZG01,LZG02,LZG03,LZG04
from problem.utils import get_data
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb

from old_relation.RCS_old import RCS



"""
测试日期：2024.1.26

基于old Relation 模型测试：
1. 测试FitnessCriteriaGenerator 生成数据的一致性
2. 测试FitnessCriteriaGenerator + Trainer 的训练精度
3. 测试CategoryCriteriaGenerator + Trainer 的训练精度

测试维度 n_var = 20
测试数据量 num = 200
测试题目 LZG01,LZG02,LZG03,LZG04

5 passed
"""

def show_dict(sample_dict):
    for key, value in sample_dict.items():
        print(f"{key}:\n {value}\n")
        print("-" * 50)





def test_fitness_criteria_generator():
    for _ in range(10):
        Xs,ys = get_data(LZG03(n_var=20),num=100,method='HLS')
        fg = FitnessCriteriaGenerator()
        res = fg.do(Xs, ys,is_shuffle=False)
        RXs = res['RXs']
        Rys = res['Rys']
        Rls = res['Rls']

        base_model = None
        rcs = RCS(model=base_model, r_opt='all')
        RXs_old, Rls_old = rcs.getRelationPairs(np.array(Xs), np.array(ys))
        assert np.allclose(RXs, RXs_old)
        assert np.array_equal(Rls, Rls_old)



def test_category_criteria_trainer_new_data():
    n_var = 20
    num = 200
    for problem in [LZG01(n_var=n_var),LZG02(n_var=n_var),LZG03(n_var=n_var),LZG04(n_var=n_var)]:
        Xs,ys = get_data(problem,num=num,method='HLS')
        cg = CategoryCriteriaGenerator(cutoff=0.5)
        res = cg.do(Xs, ys,is_shuffle=False)
        RXs = res['RXs']
        Rls = res['Rls']

        base_model = xgb.XGBClassifier(eval_metric='logloss')
        rcs = RCS(model=base_model, r_opt='split',tops = 0.5)
        RXs_old, Rls_old = rcs.getRelationPairs(np.array(Xs), np.array(ys).flatten())

        # on new data
        trainer = Trainer()
        k = 10
        kf = KFold(n_splits=k, shuffle=True)
        acc_new = []
        acc_old = []

        for train_index, test_index in kf.split(RXs):
            X_train, X_test = RXs[train_index], RXs[test_index]
            y_train, y_test = Rls[train_index], Rls[test_index]

            trainer.do(X_train, y_train.flatten())
            pre_ys_new = trainer.predict(X_test, transform_back=True)
            acc_new.append(np.sum(pre_ys_new == y_test.flatten()) / len(y_test))

            rcs.fit(X_train, y_train)
            pre_ys_old = rcs.predict(X_test)
            acc_old.append(np.sum(pre_ys_old == y_test.flatten()) / len(y_test))

        assert np.mean(acc_new) >= np.mean(acc_old)
        print(f"new: {np.mean(acc_new)}")
        print(f"old: {np.mean(acc_old)}")

    

def test_category_criteria_trainer_old_data():
    num = 200
    n_var = 20
    for problem in [LZG01(n_var=n_var),LZG02(n_var=n_var),LZG03(n_var=n_var),LZG04(n_var=n_var)]:
        Xs,ys = get_data(problem,num=num,method='HLS')
        cg = CategoryCriteriaGenerator(cutoff=0.5)
        res = cg.do(Xs, ys,is_shuffle=False)
        RXs = res['RXs']
        Rls = res['Rls']

        base_model = xgb.XGBClassifier(eval_metric='logloss')
        rcs = RCS(model=base_model, r_opt='split',tops = 0.5)
        RXs_old, Rls_old = rcs.getRelationPairs(np.array(Xs), np.array(ys).flatten())

        # on old data
        trainer = Trainer()
        k = 10
        kf = KFold(n_splits=k, shuffle=True)
        acc_new = []
        acc_old = []

        for train_index, test_index in kf.split(RXs_old):
            X_train, X_test = RXs_old[train_index], RXs_old[test_index]
            y_train, y_test = Rls_old[train_index], Rls_old[test_index]

            trainer.do(X_train, y_train)
            pre_ys_new = trainer.predict(X_test, transform_back=True)
            acc_new.append(np.sum(pre_ys_new == y_test.flatten()) / len(y_test))

            rcs.fit(X_train, y_train)
            pre_ys_old = rcs.predict(X_test)
            acc_old.append(np.sum(pre_ys_old == y_test.flatten()) / len(y_test))

        assert np.mean(acc_new) >= np.mean(acc_old)
        print(f"new: {np.mean(acc_new)}")
        print(f"old: {np.mean(acc_old)}")


def test_train_model_new_data():
    n_var = 20
    num = 200
    for problem in [LZG01(n_var=n_var),LZG02(n_var=n_var),LZG03(n_var=n_var),LZG04(n_var=n_var)]:
        Xs,ys = get_data(problem,num=num,method='HLS')
        fg = FitnessCriteriaGenerator()
        res = fg.do(Xs, ys,is_shuffle=False)
        RXs = res['RXs']
        Rys = res['Rys']
        Rls = res['Rls']

        base_model = xgb.XGBClassifier(eval_metric='logloss')
        rcs = RCS(model=base_model, r_opt='all')
        RXs_old, Rls_old = rcs.getRelationPairs(np.array(Xs), np.array(ys))

        # on new data
        trainer = Trainer()
        k = 10
        kf = KFold(n_splits=k, shuffle=True)
        acc_new = []
        acc_old = []

        for train_index, test_index in kf.split(RXs):
            X_train, X_test = RXs[train_index], RXs[test_index]
            y_train, y_test = Rls[train_index], Rls[test_index]

            trainer.do(X_train, y_train)
            pre_ys_new = trainer.predict(X_test, transform_back=True)
            acc_new.append(np.sum(pre_ys_new == y_test.flatten()) / len(y_test))

            rcs.fit(X_train, y_train)
            pre_ys_old = rcs.predict(X_test)
            acc_old.append(np.sum(pre_ys_old == y_test.flatten()) / len(y_test))

        assert np.mean(acc_new) >= np.mean(acc_old)
        print(f"new: {np.mean(acc_new)}")
        print(f"old: {np.mean(acc_old)}")

def test_train_model_old_data():
    n_var = 20
    num = 200
    for problem in [LZG01(n_var=n_var),LZG02(n_var=n_var),LZG03(n_var=n_var),LZG04(n_var=n_var)]:
        Xs,ys = get_data(problem,num=num,method='HLS')
        fg = FitnessCriteriaGenerator()
        res = fg.do(Xs, ys,is_shuffle=False)
        RXs = res['RXs']
        Rys = res['Rys']
        Rls = res['Rls']

        base_model = xgb.XGBClassifier(eval_metric='logloss')
        rcs = RCS(model=base_model, r_opt='all')
        RXs_old, Rls_old = rcs.getRelationPairs(np.array(Xs), np.array(ys))

        # on old data
        trainer = Trainer()
        k = 10
        kf = KFold(n_splits=k, shuffle=True)
        acc_new = []
        acc_old = []

        for train_index, test_index in kf.split(RXs_old):
            X_train, X_test = RXs_old[train_index], RXs_old[test_index]
            y_train, y_test = Rls_old[train_index], Rls_old[test_index]

            trainer.do(X_train, y_train)
            pre_ys_new = trainer.predict(X_test, transform_back=True)
            acc_new.append(np.sum(pre_ys_new == y_test.flatten()) / len(y_test))

            rcs.fit(X_train, y_train)
            pre_ys_old = rcs.predict(X_test)
            acc_old.append(np.sum(pre_ys_old == y_test.flatten()) / len(y_test))

        assert np.mean(acc_new) >= np.mean(acc_old)
        print(f"new: {np.mean(acc_new)}")
        print(f"old: {np.mean(acc_old)}")









if __name__ == "__main__":
    test_fitness_criteria_generator()
    test_train_model_new_data()
    test_train_model_old_data()
    test_category_criteria_trainer_new_data()
    test_category_criteria_trainer_old_data() 




# def test_fitness_criteria_generator(Xs,ys):
#     fg = FitnessCriteriaGenerator()
#     res = fg.do(Xs, ys,is_shuffle=False)
#     RXs = res['RXs']
#     Rys = res['Rys']
#     Rls = res['Rls']
#     trainer = Trainer()

#     # BASELINE
#     base_model = xgb.XGBClassifier(eval_metric='logloss')
#     rcs = RCS(model=base_model, r_opt='all')
#     RXs_old, Rls_old = rcs.getRelationPairs(np.array(Xs), np.array(ys))

#     print(RXs)
#     print(RXs_old)


#     is_same = np.allclose(RXs, RXs_old)
#     print(f"RXs is same: {is_same}") 
#     is_same = np.array_equal(Rls, Rls_old)
#     print(f"Rls is same: {is_same}")



#     k = 10
#     kf = KFold(n_splits=k, shuffle=True)

#     acc_list = []
#     acc_old_list = []
#     for train_index, test_index in kf.split(RXs):


#         X_train, X_test = RXs[train_index], RXs[test_index]
#         y_train, y_test = Rls[train_index], Rls[test_index]


#         trainer.do(X_train, y_train)
#         rcs.fit(X_train, y_train)
#         pre_ys_old = rcs.predict(X_test)

#         acc_old = np.sum(pre_ys_old == y_test.flatten()) / len(y_test)
#         acc_old_list.append(acc_old)

#         pre_ys = trainer.predict(X_test, transform_back=True)
#         acc = np.sum(pre_ys == y_test.flatten()) / len(y_test)
#         acc_list.append(acc)


#     print(acc_list)
#     print(np.mean(acc_list))


#     print(acc_old_list)
#     print(np.mean(acc_old_list))


















# if __name__ == "__main__":
#     # Xs = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
#     # ys = np.array([1, 2, 3, 4, 5])

#     Xs,ys = get_data(LZG03(n_var=20),num=100,method='HLS')
#     # print(Xs.shape,ys.shape)
#     # test_fitness_criteria_generator(Xs,ys)












#     fg = FitnessCriteriaGenerator()
#     cg = CategoryCriteriaGenerator()

#     fg_res = fg.do(Xs, ys)
#     cg_res = cg.do(Xs, ys)

#     RXs = fg_res['RXs']
#     Rys = fg_res['Rys']
#     Rls = fg_res['Rls']


#     # RXs = cg_res['RXs']
#     # Rls = cg_res['Rls']

#     base_model = xgb.XGBClassifier(eval_metric='logloss')
#     rcs = RCS(model=base_model, r_opt='all')
#     XRX,YRY = rcs.getRelationPairs(np.array(Xs), np.array(ys))



#     k = 10
#     kf = KFold(n_splits=k, shuffle=True)

#     acc_list = []
#     acc_old_list = []
#     trainer = Trainer()
#     for train_index, test_index in kf.split(Xs):
#         X_train, X_test = RXs[train_index], RXs[test_index]
#         y_train, y_test = Rls[train_index], Rls[test_index]


#         trainer.do(X_train, y_train)
#         pre_ys = trainer.predict(X_test, transform_back=True)
#         acc = np.sum(pre_ys == y_test.flatten()) / len(y_test)
#         acc_list.append(acc)


#         rcs.fit(X_train, y_train)
#         pre_ys_old = rcs.predict(X_test)
#         acc_old = np.sum(pre_ys_old == y_test.flatten()) / len(y_test)
#         acc_old_list.append(acc_old)






#     avg_acc = np.mean(acc_list)
#     print(f"Average accuracy: {avg_acc}")
#     avg_acc_old = np.mean(acc_old_list)
#     print(f"Average accuracy old: {avg_acc_old}")





#     base_model = xgb.XGBClassifier(eval_metric='logloss')
#     rcs = RCS(model=base_model, r_opt='all')
#     XRX,YRY = rcs.getRelationPairs(np.array(Xs), np.array(ys))
#     trainer = Trainer()
#     k = 10
#     kf = KFold(n_splits=k, shuffle=True)

#     acc_list = []
#     acc_new_list = []
#     for train_index, test_index in kf.split(Xs):
#         X_train, X_test = XRX[train_index], XRX[test_index]
#         y_train, y_test = YRY[train_index], YRY[test_index]

#         trainer.do(X_train, y_train)
#         pre_ys_new = trainer.predict(X_test, transform_back=True)
#         acc_new = np.sum(pre_ys_new == y_test.flatten()) / len(y_test)
#         acc_new_list.append(acc_new)

#         rcs.fit(X_train, y_train)
#         pre_ys = rcs.predict(X_test)

#         acc = np.sum(pre_ys == y_test.flatten()) / len(y_test)
#         acc_list.append(acc)

#     avg_acc = np.mean(acc_list)
#     print(f"Average accuracy: {avg_acc}")

#     avg_acc_new = np.mean(acc_new_list)
#     print(f"Average accuracy new: {avg_acc_new}")



