import numpy as np

def is_positive_definite(matrix):
    """
    Check if the matrix is positive definite
    :param matrix: numpy ndarray
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The matrix is not square")
    
    if not np.allclose(matrix, matrix.T):
        raise ValueError("The matrix is not symmetric")

    if np.all(np.linalg.eigvals(matrix) > 0):
        return True
    else:
        return False

def compute_norm_excercise(vector,matrix):
    """
    Compute the norm of the vector and the matrix
    :param vector: numpy array
    :param matrix: numpy ndarray
    """
    if vector.T.shape[0] != matrix.shape[0]:
        raise ValueError("The vector is not a column vector and not is the same size")
    
    
    
    norm = np.sqrt(np.dot(vector.T,matrix).dot(vector))

    return norm
   

