import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import scipy.io
import xgboost as xgb


class RCS(object):
    def __init__(self, model, r_opt='split',tops = None):
        """
        初始化
        :param model:
        :param r_opt:
        """
        self._base_model = model

        self.opt = r_opt

        self.SS_list = []
        self.acc_list = []
        self.model_list = []
        self.train_data_bd = []
        self.good_list = []
        self.bad_list = []
        self.bound_list = []

        self.data = []

        self.tops = tops


    def convert_label(self, ys):
        # 将含有-1 的标签转换为[0,1,2,..]
        if self.opt == 'split':
            new_ys = np.zeros(shape=ys.shape, dtype=np.int32)
            new_ys[ys == -1] = 2
            new_ys[ys == 1] = 1
        elif self.opt == 'all':
            new_ys = np.zeros(shape=ys.shape, dtype=np.int32)
            new_ys[ys == 1] = 1
        else:
            pass
        return new_ys

    def deconbert_label(self, ys):
        # 逆转换
        if self.opt == 'split':
            new_ys = np.zeros(shape=ys.shape, dtype=np.int32)
            new_ys[ys == 2] = -1
            new_ys[ys == 1] = 1
        elif self.opt == 'all':
            new_ys = np.zeros(shape=ys.shape, dtype=np.int32)
            new_ys[ys == 0] = -1
            new_ys[ys == 1] = 1

        else:
            pass
        return new_ys

    def score(self, X_test, y_test):
        """
        计算模型的交叉验证score
        :param X_test:
        :param y_test:
        :return:
        """
        y_test = self.convert_label(y_test)
        c_model = self.get_current_model()
        # X_test = self.SS_list[-1].transform(X_test)
        return c_model.score(X_test, y_test)

    def fit(self, Xs, ys):
        """
         拟合关系对数据
        :param Xs:
        :param ys:
        :return:
        """

        c_model = clone(self._base_model)

        # 将数据做转换
        ys = self.convert_label(ys)

        SS = None
        # SS = StandardScaler().fit(Xs)
        # Xs = SS.transform(Xs)
        c_model.fit(Xs, ys)

        self.SS_list.append(SS)
        self.model_list.append(c_model)

    def get_current_model(self):
        """
        获取当前模型
        :return:
        """
        return self.model_list[-1]

    def predict(self, Xs):
        # Xs = self.SS_list[-1].transform(Xs)
        ys = self.model_list[-1].predict(Xs)
        new_ys = self.deconbert_label(ys)
        return new_ys

    def assigned_wegit(self, ys):
        """
        这个后续需要修改权重的映射关系
        :param ys:
        :return:
        """
        return 0.5 * np.ones(shape=ys.shape)

    def assisigend_value(self, off_Decs, prob=None):
        t_data = self.data[-1]
        # 标签数据及维度 （用于投票）
        Xs = t_data['Xs']
        ys = t_data['ys']
        c_num = Xs.shape[0]

        # 获取 预测数据维度
        off_num = off_Decs.shape[0]
        raw_dim = off_Decs.shape[1]

        # 权重
        y_wp = self.assigned_wegit(ys)
        y_wn = 1 - y_wp

        if self.opt == 'split':

            split_mask = t_data['split_mask']

            # 构造一个预测矩阵
            all_testdata = np.zeros(shape=(2 * c_num * off_num, 2 * raw_dim))
            Xs_tile = np.tile(Xs, (off_num, 1))
            # [0,Xs]
            all_testdata[:c_num * off_num, raw_dim:] = Xs_tile
            # [Xs,0]
            all_testdata[c_num * off_num:, :raw_dim] = Xs_tile
            scores = np.zeros(shape=(off_num, 1))

            for i, u in enumerate(off_Decs):
                original = i * c_num
                Ui = np.tile(u, (c_num, 1))

                # [Ui,Xs]
                all_testdata[original:original + c_num, :raw_dim] = Ui
                # [Xs,Ui]
                all_testdata[c_num * off_num + original:c_num * off_num + original + c_num, raw_dim:] = Ui

            pre_out = self.predict(all_testdata)

            # true_out = self.true_predict(all_testdata, prob)
            # print(sum(true_out == pre_out) / pre_out.shape[0])
            # pre_out = true_out


            # 计算score
            y_wp_tile = np.tile(y_wp, off_num)
            y_wn_tile = np.tile(y_wn, off_num)


            # for UX
            pre_UiX = pre_out[:c_num * off_num]
            basic_score = np.ones(shape=pre_UiX.shape)
            S_pre_UiX = np.zeros(shape=pre_UiX.shape)

            t_mask = pre_UiX == 1
            # S_pre_UiX[t_mask] = pre_UiX[t_mask] * y_wp_tile[t_mask]
            S_pre_UiX[t_mask] = basic_score[t_mask] * y_wp_tile[t_mask]

            t_mask = pre_UiX == 0
            # t_mask and split_mask
            split_mask_tile =  np.tile(split_mask, off_num)
            t_mask_and_good = [m1 and m2 for m1, m2 in zip(t_mask.flatten(), split_mask_tile.flatten())]
            t_mask_and_bad = [m1 and m2 for m1, m2 in zip(t_mask.flatten(), ~split_mask_tile.flatten())]


            # S_pre_UiX[t_mask_and_good] = pre_UiX[t_mask_and_good] * y_wp_tile[t_mask_and_good]
            # S_pre_UiX[t_mask_and_bad] = pre_UiX[t_mask_and_bad] * y_wp_tile[t_mask_and_bad]
            S_pre_UiX[t_mask_and_good] = basic_score[t_mask_and_good] * y_wp_tile[t_mask_and_good]
            S_pre_UiX[t_mask_and_bad] = -basic_score[t_mask_and_bad] * y_wp_tile[t_mask_and_bad]

            t_mask = pre_UiX == -1
            # S_pre_UiX[t_mask] = -pre_UiX[t_mask] * y_wp_tile[t_mask]
            S_pre_UiX[t_mask] = -basic_score[t_mask] * y_wp_tile[t_mask]

            # for XU
            pre_XUi = pre_out[c_num * off_num:]
            basic_score = np.ones(shape=pre_XUi.shape)
            S_pre_XUi = np.zeros(shape=pre_XUi.shape)

            t_mask = pre_XUi == 1
            # S_pre_XUi[t_mask] = -pre_XUi[t_mask] * y_wn_tile[t_mask]
            S_pre_XUi[t_mask] = -basic_score[t_mask] * y_wn_tile[t_mask]

            t_mask = pre_XUi == 0
            # t_mask and split_mask
            split_mask_tile = np.tile(split_mask, off_num)
            t_mask_and_good = [m1 and m2 for m1, m2 in zip(t_mask.flatten(), split_mask_tile.flatten())]
            t_mask_and_bad = [m1 and m2 for m1, m2 in zip(t_mask.flatten(), ~split_mask_tile.flatten())]

            # t_mask_and_good_tile = np.tile(t_mask_and_good, off_num)
            # t_mask_and_bad_tile = np.tile(t_mask_and_bad, off_num)

            # S_pre_XUi[t_mask_and_good] = pre_XUi[t_mask_and_good] * y_wp_tile[t_mask_and_good]
            # S_pre_XUi[t_mask_and_bad] = pre_XUi[t_mask_and_bad] * y_wp_tile[t_mask_and_bad]
            S_pre_XUi[t_mask_and_good] = -basic_score[t_mask_and_good] * y_wp_tile[t_mask_and_good]
            S_pre_XUi[t_mask_and_bad] = basic_score[t_mask_and_bad] * y_wp_tile[t_mask_and_bad]

            t_mask = pre_XUi == -1
            # S_pre_XUi[t_mask] = pre_XUi[t_mask] * y_wn_tile[t_mask]
            S_pre_XUi[t_mask] = basic_score[t_mask] * y_wn_tile[t_mask]

            # 统计
            # 这里是+号
            for i in range(off_num):
                scores[i] = np.sum(S_pre_UiX[i * c_num:i * c_num + c_num]) + \
                            np.sum(S_pre_XUi[i * c_num:i * c_num + c_num])
            return scores/c_num

            # # true_out2 = self.true_predict2(off_Decs, ys, prob)
            # true_out = self.true_predict(all_testdata, prob)
            # #
            # print(sum(true_out == pre_out) / pre_out.shape[0])
            # # print(sum(true_out2 == pre_out) / pre_out.shape[0])





        elif self.opt == 'all':
            # 构造一个预测矩阵
            all_testdata = np.zeros(shape=(2 * c_num * off_num, 2 * raw_dim))
            Xs_tile = np.tile(Xs, (off_num, 1))
            # [0,Xs]
            all_testdata[:c_num * off_num, raw_dim:] = Xs_tile
            # [Xs,0]
            all_testdata[c_num * off_num:, :raw_dim] = Xs_tile
            scores = np.zeros(shape=(off_num, 1))

            for i, u in enumerate(off_Decs):
                original = i * c_num
                Ui = np.tile(u, (c_num, 1))

                # [Ui,Xs]
                all_testdata[original:original + c_num, :raw_dim] = Ui
                # [Xs,Ui]
                all_testdata[c_num * off_num + original:c_num * off_num + original + c_num, raw_dim:] = Ui
            pre_out = self.predict(all_testdata)
            # true_out = self.true_predict(all_testdata, prob)
            # print(sum(true_out == pre_out) / pre_out.shape[0])
            #
            # pre_out = true_out

            # 计算score
            y_wp_tile = np.tile(y_wp, off_num)
            y_wn_tile = np.tile(y_wn, off_num)

            # for UX
            pre_UiX = pre_out[:c_num * off_num]
            MASK = pre_UiX == 1
            S_pre_UiX = np.zeros(shape=pre_UiX.shape)
            S_pre_UiX[MASK] = pre_UiX[MASK] * y_wp_tile[MASK]
            S_pre_UiX[~MASK] = pre_UiX[~MASK] * y_wn_tile[~MASK]

            # for XU
            pre_XUi = pre_out[c_num * off_num:]
            MASK = pre_XUi == 1
            S_pre_XUi = np.zeros(shape=pre_XUi.shape)
            S_pre_XUi[MASK] = pre_XUi[MASK] * y_wn_tile[MASK]
            S_pre_XUi[~MASK] = pre_XUi[~MASK] * y_wp_tile[~MASK]

            # 统计
            for i in range(off_num):
                scores[i] = np.sum(S_pre_UiX[i * c_num:i * c_num + c_num]) - \
                            np.sum(S_pre_XUi[i * c_num:i * c_num + c_num])
            return scores

        else:
            pass

    # def assisigend_value_t(self, off_Decs, prob=None):
    #     t_data = self.data[-1]
    #     Xs = t_data['Xs']
    #     ys = t_data['ys']
    #
    #     c_num = Xs.shape[0]
    #     off_num = off_Decs.shape[0]
    #     raw_dim = off_Decs.shape[1]
    #
    #     y_wp = self.assigned_wegit(ys)
    #     y_wn = 1 - y_wp
    #
    #     # 构造一个预测矩阵
    #     all_testdata = np.zeros(shape=(2 * c_num * off_num, 2 * raw_dim))
    #     Xs_tile = np.tile(Xs, (off_num, 1))
    #     # [0,Xs]
    #     all_testdata[:c_num * off_num, raw_dim:] = Xs_tile
    #     # [Xs,0]
    #     all_testdata[c_num * off_num:, :raw_dim] = Xs_tile
    #     scores = np.zeros(shape=(off_num, 1))
    #
    #     for i, u in enumerate(off_Decs):
    #         original = i * c_num
    #         Ui = np.tile(u, (c_num, 1))
    #
    #         # [Ui,Xs]
    #         all_testdata[original:original + c_num, :raw_dim] = Ui
    #         # [Xs,Ui]
    #         all_testdata[c_num * off_num + original:c_num * off_num + original + c_num, raw_dim:] = Ui
    #
    #     pre_out = self.predict(all_testdata)
    #
    #     true_out2 = self.convert_label(self.true_predict2(off_Decs, ys, prob))
    #     true_out = self.convert_label(self.true_predict(all_testdata, prob))
    #
    #
    #     print(sum(true_out == pre_out) / pre_out.shape[0])
    #     print(sum(true_out2 == pre_out) / pre_out.shape[0])

    def true_predict(self, all_testdata, prob):
        """
        获取当前预测的真实标签
        支持参数  'split' and 'all'
        :param all_testdata:
        :param prob:
        :return:
        """

        if prob == None:
            raise ValueError('Porb is None')

        # 获取 当前测试数据中的Xs 与 ys
        [num, dim] = all_testdata.shape
        X1 = all_testdata[:, :int(dim / 2)]
        X2 = all_testdata[:, int(dim / 2):]
        Y1 = np.array([prob.obj_func(x1) for x1 in X1])
        Y2 = np.array([prob.obj_func(x2) for x2 in X2])

        if self.opt == 'split':

            c_bound = self.bound_list[-1]

            L1 = np.ones(shape=Y1.shape)
            L2 = np.ones(shape=Y2.shape)

            L1[Y1 >= c_bound] = -1
            L2[Y2 >= c_bound] = -1

            real_l = np.zeros(shape=L1.shape)
            index = 0
            for l1, l2 in zip(L1, L2):
                if l1 == 1 and l2 == -1:
                    real_l[index] = 1
                elif l1 == -1 and l2 == 1:
                    real_l[index] = -1
                else:
                    real_l[index] = 0
                index += 1

        elif self.opt == 'all':
            real_l = np.zeros(shape=Y1.shape)
            index = 0
            for y1, y2 in zip(Y1, Y2):
                if y1 < y2:
                    real_l[index] = 1
                elif y1 >= y2:
                    real_l[index] = -1
                else:
                    pass
                index += 1

        return real_l.astype(np.int)

    def true_predict2(self, off_Decs, ys, prob):
        """
        获取当前预测的真实标签  完全模拟

        支持参数  'split' and 'all'
        :param off_Decs:
        :param ys:
        :param prob:
        :return:
        """
        if prob == None:
            raise ValueError('Porb is None')
        if self.opt == 'split':
            # 先构成关系对
            uys = np.array([prob.obj_func(ui) for ui in off_Decs])
            uys_l = np.ones(shape=uys.shape)
            ys_l = np.ones(shape=ys.shape)
            c_bound = self.bound_list[-1]
            uys_l[uys >= c_bound] = -1
            ys_l[ys >= c_bound] = -1

            real_l = np.zeros(shape=(2 * uys_l.shape[0] * ys_l.shape[0],))
            index = 0
            helf_size = uys_l.shape[0] * ys_l.shape[0]
            for usi in uys_l:
                for ysi in ys_l:
                    if usi == 1 and ysi == -1:
                        real_l[index] = 1
                        real_l[index + helf_size] = -1
                    elif usi == -1 and ysi == 1:
                        real_l[index] = -1
                        real_l[index + helf_size] = 1
                    else:
                        real_l[index] = 0
                        real_l[index + helf_size] = 0
                    index += 1
        elif self.opt == 'all':
            pass
        else:
            pass

        return real_l.astype(np.int)

    def split_data(self, Xs, ys, split_criteria='mean', tops=0.1):
        if self.tops != None:
            tops = self.tops
        if split_criteria == 'mean':
            mean_bound = np.mean(ys)
            self.bound_list.append(mean_bound)
            Xs_t = copy.deepcopy(Xs)
            mask = mean_bound > ys
            X_good = Xs_t[mask, :]
            X_bad = Xs_t[~mask, :]
            return X_good, X_bad, mask

        elif split_criteria == 'tops':
            ys_t = copy.deepcopy(ys)
            ys_t.sort()

            splict_bound = ys_t[int(Xs.shape[0] * tops)]
            self.bound_list.append(splict_bound)
            mask = splict_bound > ys
            X_good = Xs[mask, :]
            X_bad = Xs[~mask, :]
            return X_good, X_bad, mask
        

    # def getRelationPairs_t(self, Xs, ys):
    #     Xs, ys = np.array(Xs), np.array(ys)
    #
    #     splict_bound = self.bound_list[-1]
    #     mask = splict_bound > ys
    #     X_good = Xs[mask, :]
    #     X_bad = Xs[~mask, :]
    #
    #     C1C2 = combvec_np(X_good, X_bad)
    #     C1C1 = combvec_np(X_good, X_good)
    #     C2C1 = combvec_np(X_bad, X_good)
    #     C2C2 = combvec_np(X_bad, X_bad)
    #
    #     # 调整样本个数
    #     t_num = int(C1C2.shape[0] / 2)
    #     if C1C1.shape[0] > t_num and C2C2.shape[0] > t_num:
    #         C1C1 = C1C1[np.random.permutation(range(C1C1.shape[0]))[:t_num], :]
    #         C2C2 = C2C2[np.random.permutation(range(C2C2.shape[0]))[:t_num], :]
    #     elif C1C1.shape[0] < t_num:
    #         C2C2 = C2C2[np.random.permutation(range(C2C2.shape[0]))[:2 * t_num - C1C1.shape[0]], :]
    #     elif C2C2.shape[0] < t_num:
    #         C1C1 = C1C1[np.random.permutation(range(C1C1.shape[0]))[:2 * t_num - C2C2.shape[0]], :]
    #
    #     XRX = np.concatenate((C1C1, C2C2, C1C2, C2C1), axis=0)
    #
    #     Ls = np.concatenate(
    #         (np.zeros(shape=(C1C1.shape[0], 1)), np.zeros(shape=(C2C2.shape[0], 1)),
    #          np.ones(shape=(C1C2.shape[0], 1)),
    #          -np.ones(shape=(C2C1.shape[0], 1))), axis=0)
    #
    #     return XRX, Ls.astype(np.int)

    # def getRelationPairs_all(self, Xs, ys):
    #     XRX = [np.r_[x1, x2] for x1 in Xs for x2 in Xs]
    #     YRY = [np.r_[y1, y2] for y1 in ys for y2 in ys]
    #
    #     t_ind = [ind for ind, value in enumerate(YRY) if value[0] < value[1]]
    #
    #     YRY = np.array([-1 for _ in range(len(YRY))])
    #     YRY[t_ind] = 1
    #
    #     del_ind = [i * len(ys) + j
    #                for i in list(range(len(ys))) for j in list(range(len(ys))) if i == j]
    #
    #     XRX = np.delete(XRX, del_ind, 0)  # 0 means del by row
    #     YRY = np.delete(YRY, del_ind)
    #
    #     return XRX, YRY.astype(np.int)

    def getRelationPairs(self, Xs, ys):
        """
        构建关系对
        两种组合方式:
        :param Xs:
        :param ys:
        :param opt:
        :return:
        """
        t_data = {}
        t_data['Xs'] = Xs
        t_data['ys'] = ys

        self.train_data_bd.append([np.min(Xs, axis=0), np.max(Xs, axis=0)])
        if self.opt == 'split':
            """
            三种标签：
            [x1, x2]
            x1 in good x2 in good ---> 0
            x2 in bad  x2 in bad  ---> 0
            x1 in good x2 in bad  ---> +1
            x1 in bad  x2 in good ---> -1 
            
            """
            x_np, y_np = np.array(Xs), np.array(ys)
            X_good, X_bad, split_mask = self.split_data(x_np, y_np, 'tops',self.tops)

            t_data['X_good'] = X_good
            t_data['X_bad'] = X_bad
            t_data['split_mask'] = split_mask

            C1C2 = combvec_np(X_good, X_bad)
            C1C1 = combvec_np(X_good, X_good)
            C2C1 = combvec_np(X_bad, X_good)
            C2C2 = combvec_np(X_bad, X_bad)

            # 调整样本个数
            t_num = int(C1C2.shape[0] / 2)
            if C1C1.shape[0] > t_num and C2C2.shape[0] > t_num:
                C1C1 = C1C1[np.random.permutation(range(C1C1.shape[0]))[:t_num], :]
                C2C2 = C2C2[np.random.permutation(range(C2C2.shape[0]))[:t_num], :]
            elif C1C1.shape[0] < t_num:
                C2C2 = C2C2[np.random.permutation(range(C2C2.shape[0]))[:2 * t_num - C1C1.shape[0]], :]
            elif C2C2.shape[0] < t_num:
                C1C1 = C1C1[np.random.permutation(range(C1C1.shape[0]))[:2 * t_num - C2C2.shape[0]], :]

            XRX = np.concatenate((C1C1, C2C2, C1C2, C2C1), axis=0)

            Ls = np.concatenate(
                (np.zeros(shape=(C1C1.shape[0], 1)), np.zeros(shape=(C2C2.shape[0], 1)),
                 np.ones(shape=(C1C2.shape[0], 1)),
                 -np.ones(shape=(C2C1.shape[0], 1))), axis=0)

            self.data.append(t_data)
            return XRX, Ls
        elif self.opt == 'all':
            """
            两种标签: 
            [x1, x2]
            x1 < x2  ---> +1
            x1 >= x2 ---> -1
            """
            XRX = [np.r_[x1, x2] for x1 in Xs for x2 in Xs]
            YRY = [np.r_[y1, y2] for y1 in ys for y2 in ys]

            t_ind = [ind for ind, value in enumerate(YRY) if value[0] < value[1]]

            YRY = np.array([-1 for _ in range(len(YRY))])
            YRY[t_ind] = 1

            del_ind = [i * len(ys) + j
                       for i in list(range(len(ys))) for j in list(range(len(ys))) if i == j]

            XRX = np.delete(XRX, del_ind, 0)  # 0 means del by row
            YRY = np.delete(YRY, del_ind)

            self.data.append(t_data)

            return XRX, YRY
        else:
            raise ValueError('{} is not support!'.format(self.opt))


def combvec_np(A_matrix, B_matrix):
    a_l, a_d = A_matrix.shape
    b_l, b_d = B_matrix.shape
    res_matrix = np.zeros(shape=(a_l * b_l, 2 * a_d))
    for i, a in enumerate(A_matrix):
        a_tile = np.tile(a, (b_l, 1))
        res_matrix[i * b_l:(i + 1) * b_l, :] = np.concatenate((a_tile, B_matrix), axis=1)
    return res_matrix
