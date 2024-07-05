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
    
    def loss_function_augmented(self, X, y, beta_augmented):
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
        beta_augmented = np.sqrt(1+self.lambda_l2_penality) * beta_augmented
        
        mse = np.mean((y - X.dot(beta_augmented))**2)

        l1_penality = self.gamma * np.sum(np.abs(beta_augmented)) 

        cost = mse + l1_penality 
        
        return cost
    
    def gradient_step(self, X, y, beta, mu):
        """
        Compute the gradient step for Lasso Regression
        z = beta + 2 * mu * X^T * (y - X * beta)

        Parameters:
        - X: matrix
        - y: ndarray
        - beta: parameters shape (n_features,) or n_features+1 (bias included)
        - mu: scalar

        return: z ndarray
        """        
       
        z = beta +  mu * X.T @ (y - X @ beta)/X.shape[0]
        
        return z
    
    def proximal_step(self, z, mu_lambda):
        """
        Compute the proximal step for Lasso Regression
        prox(z) = sign(z) * max(|z| - mu * lambda / 2, 0)

        Parameters:
        - z: ndarray
        - mu: scalar
        - lambda_l1_penality: parameter of l1 regularization

        return: update_beta ndarray
        """
       
        
        result = np.zeros_like(z)  # Creamos un array de ceros con la misma forma que z
    
        # Aplicamos las condiciones directamente usando operaciones vectorizadas de NumPy
        mask1 = (z > mu_lambda)
        mask2 = (z < -mu_lambda)
    
        result[mask1] = z[mask1] - mu_lambda
        result[mask2] = z[mask2] + mu_lambda
    
        return result

    
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
        X, y = self.data_augmentation(X,y)
         # initial values
        beta = np.zeros(X.shape[1])

        # parameter L: Lipschitz constant mu = 1/L, L may be max eigenvalue of (X.T,X)

        L = np.max(np.linalg.eigvals(np.dot(X.T, X)))
            
        mu = 1 / L

        #lambda_l1_penality = gamma, obteined through lambda1/sqrt(1+lambda2)
        
        mu_lambda = self.gamma * mu


        #cost_previous = self.loss_function_augmented(X, y, beta)
        for i in range(self.max_iterations):
           
            z = self.gradient_step(X, y, beta, mu)
            
            update_beta =  self.proximal_step(z, mu_lambda)

            #cost_current = self.loss_function(X, y, update_beta)
            

            if np.linalg.norm(update_beta-beta)  < self.threshold:
                print(f"Optimal coefficients founded in {i} iterations")
                break
            
            #cost_previous = cost_current
            beta =  update_beta
       
        self.coef_ =  (1/np.sqrt(1+self.lambda_l2_penality)) * beta
    #def fit(self, X, y):
    #    """
    #    fitting the model according the transformation the problem
    #    return self.coef_.
    #    """
    #    X, y = self.data_augmentation(X,y)
       
    #    beta = np.zeros(X.shape[1])
        #lambda_l1_penality = gamma, obteined through lambda1/sqrt(1+lambda2)
    #    model_lasso = LassoRegression(lambda_l1_penality = self.gamma, threshold=self.threshold, max_iterations=self.max_iterations)

    #    model_lasso.fit(X, y)

    #    beta_lasso = model_lasso.coef_
        
        # 1/sqrt(1+lambda_l2_penality) * beta_lasso  )
    #    self.coef_ =  (1/np.sqrt(1+self.lambda_l2_penality)) * beta_lasso 
    
    def predict(self, X):
        """
        Compute the prediction for the model
        """
        return X.dot(self.coef_)
    

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        


    