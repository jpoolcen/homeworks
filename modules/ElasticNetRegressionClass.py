import numpy as np
from modules.NaiveElasticRegressionClass import NaiveElasticRegression

class ElasticNetRegression:
    def __init__(self,lambda_l1_penality,lambda_l2_penality,threshold=0.01,max_iterations=1000):
        self.lambda_l1_penality = lambda_l1_penality
        self.lambda_l2_penality = lambda_l2_penality
        self.max_iterations = max_iterations
        self.threshold = threshold 
        self.coef_ =None
    
    def fit(self,X,y):
        
        naive_model = NaiveElasticRegression(self.lambda_l1_penality,self.lambda_l2_penality,self.threshold,self.max_iterations)
        
        naive_model.fit(X,y)
        
        naive_model_coefficients = naive_model.coef_

    
        #equation 12 (1+lambda2) * beta(naive elastic net)

        self.coef_ = (1+self.lambda_l2_penality) * naive_model_coefficients
    
    def predict(self, X):
        """
        Compute the prediction for the model
        """
        return X.dot(self.coef_)
    

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self







