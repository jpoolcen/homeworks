import numpy as np

class LassoRegression:
    def __init__(self, lambda_l1_penality,threshold=0.001, max_iterations=1000):
        self.lambda_l1_penality = lambda_l1_penality
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.coef_ = None
        
        
    def loss_function(self, X, y, beta):
        """
        Compute the loss function for Lasso Regression
        L(lambda, beta) = MSE + lambda * |beta|_1

        Parameters:
        - X: matrix
        - y: ndarray
        - beta: parameters shape (n_features,) or n_features+1 (bias included)
        - lambda_l1_penality: penalidad lambda

        return: cost scalar
        """
        mse = np.mean((y - X.dot(beta))**2)
        
        l1_penality = self.lambda_l1_penality * np.sum(np.abs(beta))
        
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
        #z = beta + 2 * mu * X.T @ (y - X @ beta)
        #z = beta + 2 * mu * np.dot(X.T, mse) # if loss function is mean, then the value of 2 is omitted

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

    
    def fit(self, X, y):
        """
        Fit the model using Lasso Regression with SGD
        parameters:
        X: matrix of features
        y: array of response
        return:
        beta: coefficients of the model
        """
        # initial values
        beta = np.zeros(X.shape[1])

        # parameter L: Lipschitz constant mu = 1/L, L may be max eigenvalue of (X.T,X)

        L = np.max(np.linalg.eigvals(np.dot(X.T, X)))
            
        mu = 1 / L
        

        mu_lambda = self.lambda_l1_penality * mu


        cost_previous = self.loss_function(X, y, beta)
        
        for i in range(self.max_iterations):
           
            z = self.gradient_step(X, y, beta, mu)
            
            update_beta = self.proximal_step(z, mu_lambda)

            cost_current = self.loss_function(X, y, update_beta)
            

            if np.linalg.norm(update_beta-beta)  < self.threshold:
                #print(f"Optimal coefficients founded in {i} iterations")
                break
            
            cost_previous = cost_current
            beta = update_beta
        self.coef_ = beta
            
        
    
    def predict(self, X):
        """
        Predict the target values
        X: matrix
        return y: predict values

        """
        return X.dot(self.coef_)

    
    def get_params(self, deep=True):
        return {'lambda_penality': self.lambda_l1_penality}
    
    