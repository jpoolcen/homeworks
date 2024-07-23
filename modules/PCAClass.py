# load libraries
import numpy as np
import pandas as pd

class PcaClass:
    def __init__(self,X,y, n_components,feature_names):
        """
        parameters:
        X: numpy array of shape (n, d), the data matrix
        y: numpy array of shape (n,), the target vector
        n_components: int, the number of principal components to keep
        is_centered: boolean, whether the data is centered or not
        """
        self.X             = X # data matrix 
        self.y             = y  # target vector
        self.n_components  = n_components # number of principal components to keep
        self.feature_names = feature_names # list of strings, the names of the features
        self.eigenvalues   = None  # eigenvalues d elements sorted
        self.eigenvectors  = None # weight matrix dxd  all d eigenvectors sorted
        self.features_eigenvalues = None # features names and eigenvalues d elementos feature:eigenvalue
        self.features_scores = None # features names and scores d elementos feature:score
        self.W_pca         = None # data matrix in the PCA space nxd
        self.W_mean_class = None # mean of the features for each class matrix 2xd
        self.scores = None   # scores of the features 
        self.S_k = None  #features selected by score
        self.features_selected = None # selected features by score
       
    

    def compute_covariance_matrix(self,X):
        """
        Compute the covariance matrix of the data.
        parameters:
        X: numpy array of shape (n, d), the data matrix
        returns: cov: numpy array of shape (d, d), the covariance matrix
        """
        return np.dot(X.T, X) / X.shape[0]


    def sorted_features_by_eigenvalues(self,eigen_values,eigen_vectors,feature_names):
        """
        function to sort the features by eigenvalues
        parameters:
        eigen_values: numpy array of shape (d,), the eigenvalues
        eigen_vectors: numpy array of shape (d, d), the eigenvectors
        feature_names: list of strings, the names of the features
        returns: sorted_features: list of strings, the names of the features sorted by eigenvalues
        """
       
        self.features_eigenvalues = dict(zip(self.feature_names  ,eigen_values))
        #sorted features names by eigenvalues
        self.features_eigenvalues = dict(sorted(self.features_eigenvalues.items(), key=lambda item: item[1], reverse=True))


        eigen_values = np.sort(eigen_values,)[::-1]  
        
        idx = np.argsort(eigen_values)[::-1]
        
        eigen_vectors = eigen_vectors[:, idx]
        
        return eigen_values, eigen_vectors

    def compute_projections(self, X):
        """Project the data matrix X onto the PCA space.
        parameters:
        X: numpy array of shape (n, d), the data matrix, X is centered
        eigenvectors: numpy array of shape (d, d), the weight matrix
        returns: X_pca: numpy array of shape (n, d), the data matrix in the PCA space
        """
        
        return np.dot(X, self.eigenvectors)   

    def compute_mean_feature_by_class(self, W,y):
        """Compute the mean of the features for each class. equation 13
        parameters:
        W: numpy array of shape (d, d), the weight matrix
        y: numpy array of shape (n, ), the class labels
        returns: means: numpy array of shape (n_classes, d), the mean of the features for each class
        """

        W_c= np.zeros((len(np.unique(y)),W.shape[1]))
        for i,c in enumerate(np.unique(y)):
            # Compute the numerator and denominator of the equation
            numerator = np.sum(W[np.where(y == c)], axis=0)
            denominator = np.sum(y == c)

            # Compute the weight for class c and feature i
            W_c[i] = numerator / denominator
        

        return W_c

    def compute_score_feature(self):
        """Compute the score of the features.
        parameters:
        W_means: numpy array of shape (n_classes, d), the mean of the features for each class
        
        eigen_values: numpy array of shape (d, ), the eigenvalues

        returns: scores: numpy array of shape (d, ), the score of the features
        """
        self.scores = np.zeros(self.W_mean_class.shape[1])

        difference =np.abs(self.W_mean_class[1]-self.W_mean_class[0])

        for i in range(difference.shape[0]):
            if self.eigenvalues[i] == 0:
                self.scores[i] = 0
            else:
                self.scores[i] = difference[i]/self.eigenvalues[i]
        
        #return scores
    def select_top_features(self):
        """Select the top n_components features based on the score.
        returns: top_features: dict, the dictionary of the top n_components features
        """
        items = list(self.features_scores.items())
        self.features_selected = dict(items[:self.n_components])
        
    
    def select_features_by_score(self):
        """
        Select the features based on the score.
        returns: 
            -selected_features: list of strings, the names of the selected features.
            -S_k: numpy array of shape (d, n_components), the selected features

        """
        # sorted orginal features names by eigenvalues
        sorted_features_by_eigenvalues = dict(sorted(self.features_eigenvalues.items(), key=lambda item: item[1], reverse=True))

        # assign the scores to the features
        items = list(sorted_features_by_eigenvalues.items())
        self.features_scores = {}
        for i in range(len(items)):
            key, value = items[i]
            self.features_scores[key] = self.scores[i]
        
        # sort the features by scores
        
        idx = np.argsort(self.scores)[::-1]

        # Use these indices to reorder the columns of the eigenvectors matrix
    
        sorted_eigen_vectors = self.eigenvectors[:, idx]

        # Select the first n_components columns of the sorted eigenvectors matrix according of score
        self.S_k = sorted_eigen_vectors[:, :self.n_components]

        #sorted features names by score
        self.features_scores = dict(sorted(self.features_scores.items(), key=lambda item: item[1], reverse=True))


        return self.features_scores,self.S_k

    def fit(self):
        """
        Main method to fit the PCA model. 
        Use the steps of the PCA algorithm and select the features based on the score
        returns: X_projected: numpy array of shape (n, n_components), the data matrix in the PCA space
        """
        
        

        # step 1: matrix of covariance of the data dxd
        matrix_covariance = self.compute_covariance_matrix(self.X)

        # step 2: eigenvalues and eigenvectors
        eig_values, eig_vectors = np.linalg.eig(matrix_covariance)

        # step 3: sort the eigenvectors by decreasing eigenvalues

        self.eigenvalues, self.eigenvectors = self.sorted_features_by_eigenvalues(eig_values, eig_vectors, self.feature_names)

        # step 4: project the data onto the PCA space
        self.W_pca = self.compute_projections(self.X)

        # step 5: compute the mean of the features for each class. step 2 of the algorithm
        self.W_mean_class = self.compute_mean_feature_by_class(self.W_pca,self.y)

        # step 6: compute the score of the features. step 3 of the algorithm

        self.compute_score_feature()
        
        # step 7: select the features based on the score. step 4 of the algorithm

        self.select_features_by_score()

        self.select_top_features()

        # step 8: return the projected data and the selected features

        X_projected = np.dot(self.X, self.S_k)

        return X_projected




        return self

    def transform(self, X):
        """
        Project the data matrix X onto the PCA space.
        parameters:
        X: numpy array of shape (n, d), the data matrix
        return X_transformed: numpy array of shape (n, n_components), the data matrix in the PCA space
        """
        
        X_transformed = np.dot(X, self.S_k)
        return X_transformed

    