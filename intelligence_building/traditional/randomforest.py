from .base import BaseModule
from sklearn.ensemble import RandomForestClassifier
from ..metrics.score import out_of_distribution_score

__all__ = ['RandomForest']
class RandomForest(BaseModule):
    
    

    """
    RandomForest Module
    use GridSearch to find the best parameters
    default params_space:
    {
        'max_depth': [10,20,30,40,50,None],
        'n_estimators': [50,100,200,300,400,500]
    }
    """


    def __init__(self,params_space=None):
        if params_space:
            pass
        else:
            params_space = {
                'max_depth': [10,20,30,40,50,None],
                'n_estimators':[50,100,200,300,400,500]
            }
        super(RandomForest,self).__init__(RandomForestClassifier(oob_score=True),params_space)


    def fit(self,X,y):
        """
        5-fold cross validation and refit the model using best parameters
        use self.svm to access the best_estimators_
        """
        super(RandomForest,self).fit(X,y)
        self.rf = self.get_best_estimators_()


    def score(self,X,y,mask):
        """
        compute the out_of_distribution_score of X,y with mask
        else compute the out_of_distribution_score for given mask
        """

        return out_of_distribution_score(self.rf.predict(X),y,mask)



    def predict(self,X):
        return self.rf.predict(X)


    def predict_proba(self,X):
        return self.rf.predict_proba(X)