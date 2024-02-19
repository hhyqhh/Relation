import xgboost as xgb






class ModelFactory:
    @staticmethod
    def get_model(model_name=None):
        if model_name is None:
            model = ModelFactory.get_default_model()
        else:
            model = ModelFactory.get_model_by_name(model_name)

        return model
            

    @staticmethod
    def get_default_model():
         return xgb.XGBClassifier(eval_metric='logloss')



    def get_model_by_name(model_name):
        raise NotImplementedError("the model is not implemented")