import numpy as np

def gen_data4voting(Xs, Us):
    """
    产生预测关系矩阵
    :param Xs:
    :param Us:
    :return:
    """
    # get shape of Xs and Us
    x_num, x_dim = Xs.shape
    u_num, u_dim = Us.shape

    if x_dim != u_dim:
        raise ValueError("Xs and Us must have the same dim")

    r_matrix = np.zeros(shape=(x_num * u_num * 2, x_dim * 2))
    Xs_tile = np.tile(Xs, (u_num, 1))
    r_matrix[:x_num * u_num, x_dim:] = Xs_tile
    r_matrix[x_num * u_num:, :x_dim] = Xs_tile

    for i, u in enumerate(Us):
        start_index = i * x_num
        gap = x_num * u_num
        Ui = np.tile(u, (x_num, 1))
        # [Ui,Xs]
        r_matrix[start_index:start_index + x_num, :x_dim] = Ui
        # [Xs,Ui]
        r_matrix[start_index + gap:start_index + gap + x_num, x_dim:] = Ui
    return r_matrix



def combvec_np(A, B):
    # get A.shape, B.shape
    a_l, a_d = A.shape
    b_l, b_d = B.shape

    if a_d != b_d:
        raise ValueError('A.shape[1] must equal to B.shape[1]')
    res = np.zeros((a_l * b_l, a_d + b_d))
    for i, a in enumerate(A):
        a_tile = np.tile(a, (b_l, 1))
        res[i * b_l:(i + 1) * b_l, :] = np.c_[a_tile, B]
    return res


def label_balance(P_0, N_0, t_num):
    """
    标签平衡
    """
    t_num = int(t_num / 2)

    if P_0.shape[0] > t_num and N_0.shape[0] > t_num:
        # random sample t_num from P_0 and N_0
        P_0 = P_0[np.random.choice(P_0.shape[0], t_num, replace=False), :]
        N_0 = N_0[np.random.choice(N_0.shape[0], t_num, replace=False), :]
    elif P_0.shape[0] < t_num:
        # random sample 2*t_num-P_0.shape[0] from N_0
        N_0 = N_0[np.random.choice(N_0.shape[0], 2 * t_num - P_0.shape[0], replace=False), :]
    elif N_0.shape[0] < t_num:
        # random sample 2*t_num-N_0.shape[0] from P_0
        P_0 = P_0[np.random.choice(P_0.shape[0], 2 * t_num - N_0.shape[0], replace=False), :]

    return P_0, N_0