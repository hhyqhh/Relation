from sklearn.base import BaseEstimator, ClassifierMixin
from relation_model.component.generator import FitnessCriteriaGenerator,CategoryCriteriaGenerator
from relation_model.component.user import FitnessCriteriaUser,CategoryCriteriaUser
from relation_model.component.trainer import Trainer
from relation_model.component.user import LeagueScoringUser
from problem.utils import get_data
from problem.single.LZG import LZG01,LZG02,LZG03,LZG04
from sklearn.naive_bayes import GaussianNB

class BaseModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y=None):
        raise NotImplementedError
    
    def real_predict(self, X, real_f):
        raise NotImplementedError



class FitnessCriteria_RSO(BaseModel):

    def __init__(self):
        self.generator = FitnessCriteriaGenerator()
        self.trainer = Trainer()
        self.user = FitnessCriteriaUser(generator=self.generator,trainer=self.trainer)

    def fit(self, X, y=None):
        res = self.generator.do(X,y)
        RXs = res['RXs']
        Rls = res['Rls']
        self.trainer.do(RXs,Rls)

    def predict(self, Us):
        return self.user.do(Us)
    
    def predict_by_real(self, Us, real_f):
        return self.user.real_do(Us, real_f)
    
class CategoryCriteria_RSO(BaseModel):

    def __init__(self):
        self.generator = CategoryCriteriaGenerator()
        self.trainer = Trainer()
        self.user = CategoryCriteriaUser(generator=self.generator,trainer=self.trainer)

    def fit(self, X, y=None):
        res = self.generator.do(X,y)
        RXs = res['RXs']
        Rls = res['Rls']
        self.trainer.do(RXs,Rls)

    def predict(self, Us):
        return self.user.do(Us)
    
    def predict_by_real(self, Us, real_f):
        return self.user.real_do(Us, real_f)
    

class RelationPreselection(BaseModel):
    def __init__(self):
        self.generator = FitnessCriteriaGenerator()
        self.trainer = Trainer(GaussianNB())
        self.user = LeagueScoringUser(generator=self.generator,trainer=self.trainer)

    def fit(self, X, y=None):
        res = self.generator.do(X,y)
        RXs = res['RXs']
        Rls = res['Rls']
        self.trainer.do(RXs,Rls)

    def predict(self, Us):
        return self.user.do(Us)
    
    def predict_by_real(self, Us, real_f):
        raise NotImplementedError

if __name__ == "__main__":
    n_var = 20

    Xs,ys = get_data(LZG01(n_var=n_var),num=100,method='HLS')
    # model = CategoryCriteria_RSO()
    # model.fit(Xs,ys)
    # Us,tys = get_data(LZG01(n_var=n_var),50)
    # pred_ys = model.predict(Us)
    # print(pred_ys)

    # model = FitnessCriteria_RSO()
    # model.fit(Xs,ys)
    # Us,tys = get_data(LZG01(n_var=n_var),50)
    # pred_ys = model.predict(Us)
    # print(pred_ys)

    model = RelationPreselection()
    model.fit(Xs,ys)
    Us,tys = get_data(LZG01(n_var=n_var),3)
    pred_ys = model.predict(Us)
    print(pred_ys)