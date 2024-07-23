import numpy as np 

def rotation_matrix(theta):
    """
    Generates a 2x2 rotation matrix based on the rotation angle
    - Parameters:
     theta (float): The rotation angle in degrees.
    -return:
     numpy.ndarray: The rotation matrix corresponding to theta.
    """
    
    #convert angle to radians
    theta_rad = np.radians(theta)

    #compute the values of the rotation matrix

    cos_theta = np.cos(theta_rad)
    sin_theta = np.cos(theta_rad)

    # build the matrix
    return np.array([[cos_theta, -sin_theta],[sin_theta,cos_theta]])


def scaling_matrix(sx,sy):
    """
    Generate a 2x2 scaling matrix based in sx and sy
    - parameters:
        sx (float): scale factor for axis x
        sy (float): scale factor for axis y
    
    - return: 
        numpy.ndarray: a 2x2 scaling matrix 
    """
    return np.array([[sx,0],[0,sy]])

def shearing_matrix(h):
    """
    Generates a 2x2 matrix shearing 

    - Parameters:
        h (float): shear factor
    - Return
        numpy.ndarray: The 2x2 shear matrix 

    """

    return np.array([[1,h],[0,1]])

def compute_matrix_transformation(R,S,H):
    """
    Compute an A matrix, when A= R.S.H.
    -Parameters:
        R (np.ndarray): Matrix rotation
        S (np.ndarray): Matrix Scaling
        H (np.ndarray): Matrix Shearing
    
    - Return:
        np.ndarray
    """
    return np.matmul(H,np.matmul(S,R))



def get_matrix_augemented(A,b):
    """
    Create a matrix with A and b
    - Parameters:
        A (ndarray): A linear transformation matrix 2x2
        b (ndarray): The translation vector (2x1)
    - Returns:
        ndarray: The matrix with A and b
    """
    A_augmented = np.hstack((A,b))
    A_augmented = np.vstack((A_augmented,[0,0,1]))
    return A_augmented 

def get_point_homogenous(point):
    """
    Create a homogenous point
    - Parameters:
        point (ndarray): The point coordinates (2x1)
    - Returns:
        ndarray: The homogenous point
    """
    return np.vstack((point,1))

def compute_affine(A, b, point):
    """
    Compute the affine transformation
    - Parameters:
        A (ndarray): A linear transformation matrix 2x2
        b (ndarray): The translation vector (2x1)
        point (ndarray): The point coordinates (2x1)
    - Returns:
        ndarray: The transformed point, row[0] = x, row[1] = y
    """
    point_homogeneous = get_point_homogenous(point)
    A_augmented = get_matrix_augemented(A,b)
   
    transformed_point_homogeneous = np.dot(A_augmented,point_homogeneous)
    return transformed_point_homogeneous


def fx(x1, x2, x3):
    """
    compute the function f(x1, x2, x3) = 10*x1**2 - x1*x2 - 5*x1*x3 + 5*x1 + 10*x2**2 - 11*x2*x3 - 2*x2 - 5*x3*x1 - 11*x3*x2 - 4*x2 + 6*x3 + 9
    parameters:
        x1 (array): float
        x2 (array): float
        x3 (array): float

    returns:array
    """
    return 10*x1**2 - x1*x2 - 5*x1*x3 + 5*x1 + 10*x2**2 - 11*x2*x3 - 2*x2 - 5*x3*x1 - 11*x3*x2 - 4*x2 + 6*x3 + 9

def center_data(X):
    """
    function to center the data. X-X.mean(axis=0)
    parameters: X: numpy array of shape (n, d), the data matrix
    returns: X: numpy array of shape (n, d), the data matrix
    """
    return X - np.mean(X, axis=0)