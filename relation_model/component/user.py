from loguru import logger
import yaml
import os
import numpy as np
import warnings

def check_compatibility(user, generator, rules_path=None):
    """
    Check if the generator is compatible with the user
    """
    if rules_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(base_dir, "compatibility_rules.yaml")
    with open(rules_path) as f:
        rules = yaml.safe_load(f)

    user_name = user.__class__.__name__
    generator_name = generator.__class__.__name__
    user_rules = rules.get(user_name,{})
    if generator_name not in user_rules.get('Generator', []):
        return False
    else:
        return True




class User:
    def __init__(self, generator=None, trainer=None):
        if generator is None:
            raise ValueError("generator is None")
        if trainer is None:
            raise ValueError("trainer is None")
        if not check_compatibility(self,generator):
            raise ValueError("generator is not compatible with user")
        self.generator = generator
        self.trainer = trainer

    def do(self,Us,return_index=False):
        Uys = self._do(Us)
        if return_index:
            return Uys,np.argsort(-Uys.flatten())
        else:
            return Uys
        
    def _do(self,Us):
        """
        1. 将Us 与 Xs 组成 关系对
        2. 预测数据
        3. 处理结果 
        """
        raise NotImplementedError
    
    def real_do(self,Us,real_f,return_index=False):
        Tys = self._real_do(Us,real_f)
        if return_index:
            return Tys,np.argsort(-Tys.flatten())
        else:
            return Tys

    def _real_do(self,Us,real_f):
        raise NotImplementedError
        


class FitnessCriteriaUser(User):
    def __init__(self, generator=None, trainer=None):
        super().__init__(generator, trainer)    


    def _do(self,Us):
        """
        1. 将Us 与 Xs 组成 关系对
        2. 预测数据
        3. 处理结果 
        """
        Xs, _, _ = self.generator.get_data()

        # shape of Us
        u_num, u_dim = Us.shape
        # shape of Xs
        x_num, x_dim = Xs.shape

        # get the relation
        Rs = self.generator.get_predict_relation(Us)

        # predict the ls  (voting step)
        ls = self.trainer.predict(Rs)

        # scorning step
        scores = np.empty(shape=(u_num, 1))
        reshaped_ls = ls.reshape((u_num*2,-1))

        pred_UiXs, pred_XsUi = np.split(reshaped_ls,2,axis=0)

        scores = (np.count_nonzero(pred_UiXs == 1, axis=1)
                  + np.count_nonzero(pred_XsUi == -1, axis=1)
                  - np.count_nonzero(pred_UiXs == -1, axis=1)
                  - np.count_nonzero(pred_XsUi == 1, axis=1))
        
        scores = scores / (x_num * 2)
        return scores


    def _real_do(self,Us,real_f):
        warnings.warn("real_do is used by debug, it is not recommended to use it in production environment")

        Xs,ys,_ = self.generator.get_data()
        # shape of Us
        u_num, u_dim = Us.shape
        # shape of Xs
        x_num, x_dim = Xs.shape

        # get the relation
        Rs = self.generator.get_predict_relation(Us)

        # predict by real_f
        Rs_I = np.split(Rs,2,axis=1)[0]
        Rs_II = np.split(Rs,2,axis=1)[1]

        # calculate the real function
        ys_I = real_f(Rs_I)
        ys_II = real_f(Rs_II)

        # combine ys_I and ys_II
        ys = np.concatenate((ys_I, ys_II), axis=1)
        
        # calculate the ls
        superior_mask = [ind for ind, value in enumerate(ys) if value[0] < value[1]]
        ls = np.array([-1 for _ in range(len(ys))])
        ls[superior_mask] = 1

        # scorning step
        scores = np.empty(shape=(u_num, 1))
        reshaped_ls = ls.reshape((u_num*2,-1))

        pred_UiXs, pred_XsUi = np.split(reshaped_ls,2,axis=0)

        scores = (np.count_nonzero(pred_UiXs == 1, axis=1)
                  + np.count_nonzero(pred_XsUi == -1, axis=1)
                  - np.count_nonzero(pred_UiXs == -1, axis=1)
                  - np.count_nonzero(pred_XsUi == 1, axis=1))
        
        scores = scores / (x_num * 2)
        return scores




class CategoryCriteriaUser(User):
    def __init__(self, generator=None, trainer=None):
        super().__init__(generator, trainer)


    def _do(self,Us):
        split_res = self.generator.get_split_result()
        Xs, ys, p_mask, n_mask = split_res['Xs'], split_res['ys'], split_res['p_mask'], ~split_res['p_mask']

        # shape of Us
        u_num, u_dim = Us.shape
        # shape of Xs
        x_num, x_dim = Xs.shape

        # get the relation
        Rs = self.generator.get_predict_relation(Us)

        # predict the ls  (voting step)
        ls = self.trainer.predict(Rs)

        # scorning step
        scores = np.empty(shape=(u_num, 1))
        reshaped_ls = ls.reshape((u_num*2,-1))

        pred_UiXs, pred_XsUi = np.split(reshaped_ls,2,axis=0)

        # 扩充mask
        p_masks = np.tile(p_mask,(u_num,1))
        n_masks = np.tile(n_mask,(u_num,1))

        # 计算得分
        scores =  np.count_nonzero(np.logical_and(pred_UiXs == 1, p_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_UiXs == 1, n_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_XsUi == -1, p_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_XsUi == -1, n_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_UiXs == 0, p_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_XsUi == 0, p_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_UiXs == -1, p_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_UiXs == -1, n_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_XsUi == 1, p_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_XsUi == 1, n_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_UiXs == 0, n_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_XsUi == 0, n_masks), axis=1)


        scores = scores / (x_num * 2)
        return scores
    
    def real_do(self,Us,real_f):
        warnings.warn("real_do is used by debug, it is not recommended to use it in production environment")

        split_res = self.generator.get_split_result()
        Xs, ys, p_mask, n_mask,split_boundary = split_res['Xs'], split_res['ys'], split_res['p_mask'], ~split_res['p_mask'],split_res['split_boundary']

        # shape of Us
        u_num, u_dim = Us.shape
        # shape of Xs
        x_num, x_dim = Xs.shape

        # get the relation
        Rs = self.generator.get_predict_relation(Us)

        # predict by real_f
        Rs_I = np.split(Rs,2,axis=1)[0]
        Rs_II = np.split(Rs,2,axis=1)[1]

        # calculate the real function
        ys_I = real_f(Rs_I)
        ys_II = real_f(Rs_II)

        # combine ys_I and ys_II
        ys = np.concatenate((ys_I, ys_II), axis=1)
        
        real_mask = ys < split_boundary

        real_mask_ux = np.split(real_mask,2,axis=0)[0]
        real_mask_xu = np.split(real_mask,2,axis=0)[1]

        # FOR UX
        ls_pred_ux = np.zeros((real_mask_ux.shape[0]))
        ls_pred_ux[np.logical_and(real_mask_ux[:, 0], np.logical_not(real_mask_ux[:, 1]))] = 1
        ls_pred_ux[np.logical_and(np.logical_not(real_mask_ux[:, 0]), real_mask_ux[:, 1])] = -1

        # FOR XU
        ls_pred_xu = np.zeros((real_mask_xu.shape[0]))
        ls_pred_xu[np.logical_and(real_mask_xu[:, 0], np.logical_not(real_mask_xu[:, 1]))] = -1
        ls_pred_xu[np.logical_and(np.logical_not(real_mask_xu[:, 0]), real_mask_xu[:, 1])] = 1

        # combine ls_pred_ux and ls_pred_xu
        ls = np.concatenate((ls_pred_ux, ls_pred_xu), axis=0)


        # scorning step
        scores = np.empty(shape=(u_num, 1))
        reshaped_ls = ls.reshape((u_num*2,-1))

        pred_UiXs, pred_XsUi = np.split(reshaped_ls,2,axis=0)

        # 扩充mask
        p_masks = np.tile(p_mask,(u_num,1))
        n_masks = np.tile(n_mask,(u_num,1))

        # 计算得分
        scores =  np.count_nonzero(np.logical_and(pred_UiXs == 1, p_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_UiXs == 1, n_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_XsUi == -1, p_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_XsUi == -1, n_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_UiXs == 0, p_masks), axis=1)
        scores += np.count_nonzero(np.logical_and(pred_XsUi == 0, p_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_UiXs == -1, p_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_UiXs == -1, n_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_XsUi == 1, p_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_XsUi == 1, n_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_UiXs == 0, n_masks), axis=1)
        scores -= np.count_nonzero(np.logical_and(pred_XsUi == 0, n_masks), axis=1)

        scores = scores / (x_num * 2)
        return scores
