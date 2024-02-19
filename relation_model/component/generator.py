from loguru import logger
import numpy as np
from relation_model.component.utils import gen_data4voting, combvec_np, label_balance
from relation_model.component.split import split_by_means, split_by_tops

class Generator:
    """
    generate the relation pairs for training
    """
    def __init__(self):
        """
        init the generator
        """
        self.Xs_list = []
        self.ys_list = []
        self.r_list = []

        self.do_count = 0

    def do(self, Xs, ys,is_shuffle=True):
        """
        do the generator
        :param Xs: the Xs
        :param ys: the ys

        :return: None
        """
        # check the Xs and ys
        if len(Xs) != len(ys):
            logger.error("the length of Xs and ys is not equal")
            return
        
        # gat relation
        relation = self._do(Xs,ys,is_shuffle)

        # save data
        self.Xs_list.append(Xs)
        self.ys_list.append(ys)
        self.r_list.append(relation)

        # update the count
        self.do_count += 1
        return  relation
    
    def _do(self, Xs, ys,is_shuffle):
        logger.error("the generator is not implemented")

    def get_data(self,t = -1):
        """
        get the data
        :return: the data
        """
        return self.Xs_list[t], self.ys_list[t], self.r_list[t]
    

    def get_predict_relation(self,Us):
        logger.error("the generator is not implemented")

    


class FitnessCriteriaGenerator(Generator):
    """
    适应度准则关系生成器
    根据输入X1, X2, y1, y2, 生成适应度准则关系
    if:
        y1 < y2: l([X1, X2]) = 1, l([X2,X1]) = -1
    else:
        l([X1, X2]) = -1, l([X2,X1]) = 1
    """
    def __init__(self):
        super().__init__()

    def _do(self, Xs, ys, shuffle=True):
        logger.debug('FitnessCriteriaGenerator._do start')
        # get relation data
        RXs = [np.r_[x1, x2] for x1 in Xs for x2 in Xs]
        Rys = [np.r_[y1, y2] for y1 in ys for y2 in ys]

        RXs = np.array(RXs)
        Rys = np.array(Rys)

        # assign relation label
        superior_mask = [ind for ind, value in enumerate(Rys) if value[0] < value[1]]
        Rls = np.array([-1 for _ in range(len(Rys))])
        Rls[superior_mask] = 1

        # remove repeat data
        repeat_mask = [i * len(ys) + j
                       for i in list(range(len(ys))) for j in list(range(len(ys))) if i == j]

        RXs = np.delete(RXs, repeat_mask, axis=0)
        Rys = np.delete(Rys, repeat_mask, axis=0)
        Rls = np.delete(Rls, repeat_mask)

        if shuffle:
            shuffle_mask = np.arange(len(RXs))
            np.random.shuffle(shuffle_mask)
            RXs = RXs[shuffle_mask]
            Rys = Rys[shuffle_mask]
            Rls = Rls[shuffle_mask]

        logger.debug('FitnessCriteriaGenerator._do end')
        return {
            'RXs': RXs.astype(np.float32),
            'Rys': Rys,
            'Rls': Rls
        }
    

    def get_predict_relation(self,Us):
        logger.debug('FitnessCriteriaGenerator.get_predict_relation start')
        Xs = self.Xs_list[-1]
        return gen_data4voting(Xs, Us)


class CategoryCriteriaGenerator(Generator):
    def __init__(self, split_strategy='tops', cutoff=0.3):
        super().__init__()
        # 分类策略
        if split_strategy not in ['tops', 'means']:
            raise ValueError('split_strategy must be one of "tops" or "means"')
        self.split_strategy = split_strategy
        self.cutoff = cutoff

        self.split_result_list = []


    def _do(self, Xs, ys, is_shuffle=True):
        """
        """
        logger.debug('CategoryCriteriaGenerator._do')
        if self.split_strategy == 'tops':
            res = split_by_tops(Xs, ys, self.cutoff)
        elif self.split_strategy == 'means':
            res = split_by_means(Xs, ys)
        else:
            raise ValueError('split_strategy must be one of "tops" or "means"')

        self.split_result_list.append(res)

        # get relation data
        Xp = res['Xp']
        Xn = res['Xn']

        X_PN = combvec_np(Xp, Xn)  # l([Xp, Xn]) = +1
        X_PP = combvec_np(Xp, Xp)  # l([Xp, Xp]) = +0
        X_NP = combvec_np(Xn, Xp)  # l([Xn, Xp]) = -1
        X_NN = combvec_np(Xn, Xn)  # l([Xn, Xn]) = -0

        # len(X_NP) always equal to len(X_PN) but len(X_NN) not equal to len(X_PP)
        # so we need to balance the label
        X_PP, X_NN = label_balance(X_PP, X_NN, X_PN.shape[0])

        RXs = np.r_[X_PN, X_PP, X_NP, X_NN]
        # finally the label of X_PP and X_NN is 0
        Rls = np.r_[np.ones(X_PN.shape[0]), np.zeros(X_PP.shape[0]), -np.ones(X_NP.shape[0]), np.zeros(
            X_NN.shape[0])].flatten()

        if is_shuffle:
            # shuffle data
            shuffle_mask = np.random.permutation(RXs.shape[0])
            RXs = RXs[shuffle_mask]
            Rls = Rls[shuffle_mask]

        return {'RXs': RXs, 'Rls': Rls.astype(np.int32)}


    def get_predict_relation(self,Us):
        logger.debug('CategoryCriteriaGenerator.get_predict_relation start')
        Xs = self.Xs_list[-1]
        return gen_data4voting(Xs, Us)
    
    def get_split_result(self,t = -1):
        return self.split_result_list[t]


def show_dict(sample_dict):
    for key, value in sample_dict.items():
        print(f"{key}:\n {value}\n")
        print("-" * 50)

if __name__ == '__main__':
    logger.remove()

    fg = FitnessCriteriaGenerator()
    cg = CategoryCriteriaGenerator()

    Xs = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    ys = np.array([1, 2, 3, 4, 5])

    logger.info('FitnessCriteriaGenerator')
    fg_res = fg.do(Xs, ys, is_shuffle=False)
    logger.info(show_dict(fg_res))

    logger.info('CategoryCriteriaGenerator')
    cg_res = cg.do(Xs, ys, is_shuffle=False)
    logger.info(show_dict(cg_res))