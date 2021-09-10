from sklearn.model_selection import GridSearchCV

class BaseModule:
    """
    base module of traditional machine learning model
    this module will use the 5-fold cross validation to select the best hyper-parameters
    use get_best_estimators_ function to get the refit model
    """
    def __init__(self,clf,params_space,cv=5):
        self.estimators = GridSearchCV(clf,param_grid=params_space,cv=cv,n_jobs=-1,verbose=4)


    def fit(self,X,y):
        self.estimators.fit(X,y)
    

    def get_best_estimators_(self):
        print(f'Best estimator has parameter: {self.estimators.best_params_}, best score is {self.estimators.best_score_}')
        return self.estimators.best_estimator_


    def get_cv_results_(self):
        """
        return a dictionary of cv_result
        
        useful key:
        mean_test_score
        std_test_score
        params
        """
        return self.estimators.cv_results_    
    