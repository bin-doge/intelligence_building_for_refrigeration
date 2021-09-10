from .base import BaseModule
from sklearn.svm import SVC
from ..metrics.score import out_of_distribution_score

__all__ = ['SVM']
class SVM(BaseModule):


    """
    SVM Module
    use GridSearch to find the best parameters
    default params_space:
    {
        'C': [1,10,20,30,40,50,60,70,80,90,100],
        'gamma': [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    }
    """


    def __init__(self,params_space=None):
        if params_space:
            pass
        else:
            params_space = {
                'C': [1,10,20,30,40,50,60,70,80,90,100],
                'gamma':[0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
            }
        super(SVM,self).__init__(SVC(),params_space)


    def fit(self,X,y):
        """
        5-fold cross validation and refit the model using best parameters
        use self.svm to access the best_estimators_
        """
        super(SVM,self).fit(X,y)
        self.svm = self.get_best_estimators_()


    def score(self,X,y,mask):
        """
        compute the accuracy of X,y if mask=None(or mask = np.ones(X.shape[0]))
        else compute the out_of_distribution_score for given mask
        """
        return out_of_distribution_score(self.svm.predict(X),y,mask)

    

    def predict(self,X):
        return self.svm.predict(X)
        