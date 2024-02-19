from loguru import logger
from sklearn import preprocessing
from sklearn.base import clone
from relation_model.component.model_factory import ModelFactory

class Trainer:
    def __init__(self,model=None) -> None:
        if model is None:
            self.model = ModelFactory.get_default_model()
        else:
            self.model = model

        # 检验模型是否实现了fit，predict方法
        if not hasattr(self.model, 'fit'):
            raise ValueError("the model must have fit method")
        if not hasattr(self.model, 'predict'):
            raise ValueError("the model must have predict method")
        
        self.label_encoder = preprocessing.LabelEncoder()
        self.model_history = []

        

    def do(self,Xs,ys):

        # transform the ys
        logger.debug("transform the ys")
        ys_trans = self.label_encoder.fit_transform(ys)   

        # fit the model
        logger.debug("fit the model")
        self.model_history.append(self._do(Xs,ys_trans))
        return self.model_history[-1]


    def _do(self, Xs, ys):
        model = clone(self.model)
        model.fit(Xs,ys)
        return model

    def predict(self, Xs,transform_back=True,return_prob=False):
        logger.debug('Trainer.predict')
        model = self.model_history[-1]

        if return_prob and hasattr(model,'predict_proba'):
            ys_pred = model.predict_proba(Xs)
            return ys_pred
        else:
            return self._predict(Xs,transform_back)

    def _predict(self, Xs,transform_back):
        model = self.model_history[-1]
        ys_pred = model.predict(Xs)
        if transform_back:
            ys_pred = self.label_encoder.inverse_transform(ys_pred)
        else:
            ys_pred = ys_pred
        return ys_pred
