from modules.LassoRegressionClass import LassoRegression
import numpy as np 

class NaiveElasticRegression:
    def __init__(self, lambda_l1_penality, lambda_l2_penality, threshold=0.001, max_iterations=1000):
        
        self.lambda_l1_penality = lambda_l1_penality

        self.lambda_l2_penality = lambda_l2_penality
        # gamma = lambda1/sqrt(1+lambda2)
        self.gamma = self.lambda_l1_penality / np.sqrt(1+self.lambda_l2_penality)

        self.threshold = threshold
        
        self.max_iterations = max_iterations
        
        self.coef_ = None
    
    def loss_function(self, X, y, beta):
        """
        Compute the loss function for Elastic Net Regression
        L(lambda, beta) = MSE + lambda * |beta|_1 + lambda * |beta|_2^2

        Parameters:
        - X: matrix
        - y: ndarray
        - beta: parameters shape (n_features,) or n_features+1 (bias included)
        - lambda_l1_penality: penalidad lambda

        return: cost scalar
        """
        mse = np.mean((y - X.dot(beta))**2)
        
        l1_penality = self.lambda_l1_penality * np.sum(np.abs(beta))
        
        l2_penality = self.lambda_l2_penality * np.sum(beta**2)
        
        cost = mse + l1_penality + l2_penality
        
        return cost
    
    def data_augmentation(self, X,y):
        """
        Transform the data according to the lemma 1 of the paper.
        X: matrix of features
        y: vector of response
        return:
        X_augmented: Matrix with X.shape[0]+p rows
        y_augmented: 
        """

        I = np.identity(X.shape[1]) 

        X_augmented = np.vstack((X, np.sqrt(self.lambda_l2_penality) * I))

        X_augmented = 1/np.sqrt(1+self.lambda_l2_penality) * X_augmented

        y_augmented = np.concatenate((y, np.zeros(X.shape[1])))

        return X_augmented, y_augmented
       

    def fit(self, X, y):
        """
        fitting the model according the transformation the problem
        return self.coef_.
        """
        X, y = self.data_augmentation(X,y)
       
        beta = np.zeros(X.shape[1])
        #lambda_l1_penality = gamma, obteined through lambda1/sqrt(1+lambda2)
        model_lasso = LassoRegression(lambda_l1_penality = self.gamma, threshold=self.threshold, max_iterations=self.max_iterations)

        model_lasso.fit(X, y)

        beta_lasso = model_lasso.coef_
        
        # 1/sqrt(1+lambda_l2_penality) * beta_lasso  )
        self.coef_ =  (1/np.sqrt(1+self.lambda_l2_penality)) * beta_lasso 
    
    def predict(self, X):
        """
        Compute the prediction for the model
        """
        return X.dot(self.coef_)
    

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        


    