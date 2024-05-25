#import jax.numpy as np
#from jax import jit
import numpy as np
class LinearRegression:
    """"
    A simple class of linear regression model.
    The model is trained by the method of least squares.

    Attributes:
    coef_ : array, shape (n_features,) 
    intercept_ : float (scalar)

    """
    def __init__(self):
        """
        Constructor to initialize the model.
        """
        self.coef_ = None
        self.intercept_ = None

    
    def fit(self, X, y):
        """
        Fit the model accoridng to the given training data. The solution is obtained by the method of least squares.
        or closed form solution.
        """
        X = np.array(X)
        y = np.array(y)
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]



    
    def predict(self, X):
        """
        Predict the target variable for the given data.
        Parameters:
            - X is a ndarray of shape (n_samples, n_features)
        Returns: _computed target values
        """
        X = np.array(X)
        return X @ self.coef_ + self.intercept_
    
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
 