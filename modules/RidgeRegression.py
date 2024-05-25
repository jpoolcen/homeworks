import jax.numpy as np
from jax import jit

class RidgeRegression:
    """"
    Class to fit a ridge regression model using the closed-form solution.
    The model is trained by the method of least squares with L2 regularization.
    """

    def __init__(self, lambda_=1.0):
        """ Constructor to initialize the model.
            lambda_ : float, default=1.0 
        """
        self.lambda_ = lambda_
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X, y):
        """
        Fit the model according to the given training data. The solution is obtained by the method of least squares.
        or closed form solution.
        Parameters:
            X : array-like, shape (n_samples, n_features)
            y : array-like, shape (n_samples,)
        returns: self object with updated coef_ and intercept_ attributes
        """
        n_samples, n_features = X.shape
        
        # Add a column of ones for the intercept term
        X = np.hstack((np.ones((n_samples, 1)), X))
        
        # Compute the coefficients using the closed-form solution
        A = np.dot(X.T, X) + self.lambda_ * np.identity(n_features + 1)
        b = np.dot(X.T, y)
        self.coef_ = np.linalg.solve(A, b)[1:]
        self.intercept_ = np.linalg.solve(A, b)[0]

    
    def predict(self, X):
        """
        Predict the target variable for the given data.
        
        Parameters:
            X : array-like, shape (n_samples, n_features)
        
        Returns: _computed target values
        """
        n_samples = X.shape[0]
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred
    
    def score(self, X, y):
        """ Compute the R^2 score of the model. 
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns: R^2 score of the model. The value of R^2 lies between 0 and 1. more near to 1 is best
        """


        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        # R^2 = 1 - (sum of squared residuals) / (total sum of squares)
        return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def mean_squared_error(self, X, y):
        """ Compute the mean squared error of the model.
        Parameters:
        - X : array-like, shape (n_samples, n_features)
        - y : array-like, shape (n_samples,)
        
        Returns: Mean squared error of the model.
        """
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return ((y - y_pred) ** 2).mean()