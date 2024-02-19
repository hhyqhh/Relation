from loguru import logger
import copy
import numpy as np


def split_by_means(Xs, ys):
    """
    mean 分类策略
    """
    logger.debug('split_by_means')
    split_boundary = np.mean(ys)
    p_mask = split_boundary > ys
    Xp = Xs[p_mask, :]
    Xn = Xs[~p_mask, :]
    return {'Xs': Xs, 'ys': ys,
            'Xp': Xp, 'Xn': Xn,
            'p_mask': p_mask, 'split_boundary': split_boundary}


def split_by_tops(Xs, ys, cutoff=0.3):
    """
    top 分类策略
    """
    ys = ys.flatten()
    logger.debug('split_by_tops, cutoff:{}'.format(cutoff))
    ys_t = copy.deepcopy(ys)
    ys_t.sort()

    split_boundary = ys_t[int(len(ys_t) * cutoff)]
    p_mask = split_boundary > ys
    Xp = Xs[p_mask, :]
    Xn = Xs[~p_mask, :]
    return {'Xs': Xs, 'ys': ys,
            'Xp': Xp, 'Xn': Xn,
            'p_mask': p_mask, 'split_boundary': split_boundary}