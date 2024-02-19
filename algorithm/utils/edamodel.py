from abc import abstractmethod, ABC
import numpy as np
import copy
# import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt



def local_search(Xs, ys):
    Xs_new = copy.deepcopy(Xs)
    TINY = 1e-50
    [No, Dim] = Xs.shape
    for r0 in range(0, No - 2):
        r1 = r0 + 1
        r2 = r0 + 2
        for d in range(Dim):
            if abs(Xs[r1, d] - Xs[r0, d]) > TINY and abs(Xs[r2, d] - Xs[r1, d]) > TINY and abs(
                    Xs[r0, d] - Xs[r2, d]) > TINY:
                a = ((ys[r1] - ys[r0]) / (Xs[r1, d] - Xs[r0, d]) - (ys[r0] - ys[r2]) / (Xs[r0, d] - Xs[r2, d])) / (
                        Xs[r1, d] - Xs[r2, d])
                b = (ys[r1] - ys[r0]) / (Xs[r1, d] - Xs[r0, d]) - a * (Xs[r1, d] + Xs[r0, d])
                if abs(a) > TINY:
                    Xs_new[r0, d] = -b / (2.0 * a)
    return Xs_new





@abstractmethod
class EDA_model(ABC):
    def __init__(self):
        pass

    def sample(self, N):
        pass

    def update(self, Xs):
        pass


class LBH(EDA_model):
    
    def __init__(self):
        super().__init__()
        self.D = None
        self.Prob = None
        self.M = None
        self.t_max = None
        self.t = 0

    def cal_gamma(self, t):
        return t / self.t_max

    def init(self, D, M, t_max):
        self.D = D
        self.M = M
        self.t_max = t_max
        self.Prob = np.ones(shape=[self.M, self.D]) / (self.M)



    def update(self, Xs):
        self.t += 1

        gamma = self.cal_gamma(self.t)
        [No, Dim] = Xs.shape
        for d in range(Dim):
            counts = np.bincount(Xs[:, d], minlength=self.M)
            col_sum = np.sum(counts) + 1e-10  # 加上一个很小的正数
            self.Prob[:, d] = (1 - gamma) * self.Prob[:, d] + gamma * counts.T / col_sum  # 计算概率分布
            self.Prob[:, d] = np.clip(self.Prob[:, d], 0, 1)



    def sample(self, N):
        col_probs = self.Prob / np.sum(self.Prob, axis=0)
        Xs = np.apply_along_axis(lambda p: np.random.choice(self.M, size=N, p=p), axis=0, arr=col_probs)

        return Xs
    


class VWH(EDA_model):
    def __init__(self, M=15):
        super().__init__()
        self.M = M
        self.D = None
        self.LB = None
        self.UB = None

        self.originalLB = None
        self.originalUB = None

        self.LB_list = None
        self.UB_list = None
        self.Prob = None
        self.Range = None

    def init(self, D, LB, UB):

        self.D = D
        self.LB = LB
        self.UB = UB

        self.originalLB = LB
        self.originalUB = UB

        self.LB_list = [LB]
        self.UB_list = [UB]
        self.Prob = np.ones(shape=[self.M, self.D]) / self.M
        self.Range = np.tile(self.LB, [self.M + 1, 1]) + np.tile(self.UB - self.LB, [self.M + 1, 1]) * np.tile(
            np.linspace(0, 1, self.M + 1).reshape(self.M + 1, -1), [1, self.D])

    def sample(self, N):
        probs = np.cumsum(self.Prob, axis=0)
        return self._sampleUMDA(probs, np.random.random([N, self.D]) * np.tile(probs[self.M - 1, :], [N, 1]),
                                np.random.random([N, self.D]), self.Range)

    def _sampleUMDA(self, Probs, prob0, prob1, ranges):
        Dim = self.D
        NM = self.M
        N = prob0.shape[0]
        pop = np.zeros(shape=prob0.shape)

        for d in range(Dim):
            for k in range(N):
                for i in range(NM):
                    if i == NM - 1:  # 这里下标是否正确？
                        index = NM - 1
                    elif prob0[k, d] <= Probs[i, d]:
                        index = i
                        break
                pop[k, d] = ranges[index, d] + prob1[k, d] * (ranges[index + 1, d] - ranges[index, d])
        return pop

    def update(self, Xs):
        NP = Xs.shape[0]  # 数据量
        Xs_t = copy.deepcopy(Xs)
        Xs_t.sort(axis=0)
        LB = Xs_t[0, :] - 0.5 * (Xs_t[1, :] - Xs_t[0, :])
        UB = Xs_t[-1, :] + 0.5 * (Xs_t[-1, :] - Xs_t[-2, :])

        # 边界检测
        mask = LB < self.originalLB
        LB[mask] = self.originalLB[mask]

        mask = UB > self.originalUB
        UB[mask] = self.originalUB[mask]

        # 更新边界
        self.UB_list.append(UB)
        self.LB_list.append(LB)
        self.LB = LB
        self.UB = UB

        self.Range[1:self.M, :] = np.tile(self.LB, [self.M - 1, 1]) + np.tile(self.UB - self.LB,
                                                                              [self.M - 1, 1]) * np.tile(
            np.linspace(0, 1, self.M - 1).reshape(self.M - 1, -1), [1, self.D])

        # for x in Xs:
        #     if (x < LB).any():
        #         print(x)
        #     if (x > UB).any():
        #         print(x)
        # update the probability vector
        index = np.floor((self.M - 2) * (Xs - np.tile(self.LB, [NP, 1])) / np.tile(self.UB - self.LB, [NP, 1]))

        if sum(np.min(index, axis=0)) < 0:
            print(np.min(index, axis=0))
            a = 1

        # 判断 index 的合理性并做修正。
        if (index > self.M - 2 - 1).any() or (np.isnan(index)).any():
            # index 大于边界的处理
            mask = index > self.M - 2 - 1
            index[mask] = self.M - 2 - 1
            # index 值为NaN 的处理
            mask = np.isnan(index)
            index[mask] = 0

        Prob = np.zeros([self.M, self.D])
        # try:
        Prob[1:self.M - 1, :] = self._UpdateUMDA(index)
        # except:
        #     print((np.isnan(index)).any())
        # #
        # #     maprintsk = index > self.M - 2 - 1
        # #     print(Xs[mask])
        # #     print(self.UB-self.LB)
        # #     print(Xs[mask]-self.LB)

        mask = self.Range[1, :] > self.Range[0, :]
        Prob[0, mask] = 0.1
        mask = self.Range[-1, :] > self.Range[-2, :]
        Prob[-1, mask] = 0.1
        Prob = Prob / np.tile(np.sum(Prob, axis=0), [self.M, 1])
        self.Prob = Prob

    def _UpdateUMDA(self, index):
        NM = self.M - 2
        Dim = self.D
        N = index.shape[0]

        Prob1 = np.ones([NM, Dim])
        for d in range(Dim):
            for k in range(N):
                Prob1[int(index[k, d]), d] += 1

        return Prob1

    def show(self, x=None):
        # sns.set_style('dark')
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), tight_layout=True)

        pop = self.sample(100000)
        ax.hist(pop, self.Range.flatten().tolist(), zorder=0)
        if x is not None:
            x = x.flatten()
            ax.scatter(x, 1000 * np.ones(shape=x.shape), c='r', zorder=1)
        plt.ylim([0, 50000])
        plt.show()



if __name__ == "__main__":
    pass
    # need to add some code for show eda model

    # need to add code for opt a function based on eda model