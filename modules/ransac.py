from sklearn.linear_model import LinearRegression
import random 
import numpy as np
def ransac(data,y,iterations,n_sample,threshold,model=LinearRegression() ):
    """
    RANSAC algorithm for robust regression.
    parameters:
        - data (ndarray): is a list of data points
        - iterations (integer positive): is the number of iterations to run
        - n_sample (integer, positive): is the number of data points to sample at each iteration
        - threshold (float): is the threshold to determine when a data point fits a model
    returns:
        - best_model: the model that best fits the data (inliers)
    """
    bestfit = None
    max_inliers = 0

    min_inliers = len(data)

    for i in range(iterations):
        indices = np.random.choice(data.shape[0], size=n_sample, replace=False)

        # Use the indices to select rows from data
        s = data[indices]
        # suppose is a linear model
        y_random = y[indices]
        #model = LinearRegression()
        model.fit(s,y_random)
        inlier_count = 0
        y_predict_model = model.predict(data)
        

        inlier_count = len(np.where(abs(y_predict_model - y) < threshold)[0])
        
        worst_fit = model.coef_
        
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            bestfit = model.coef_
            #print(f"Best model updated with {max_inliers} inliers")
        if inlier_count < min_inliers:
            min_inliers = inlier_count
            worst_fit = model.coef_
            #print(f"Min model updated with {min_inliers} inliers")
    

    print(f"Best model updated with {max_inliers} inliers")  
    print(f"Min model updated with {min_inliers} inliers")
    return bestfit,worst_fit