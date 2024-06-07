import numpy as np

def compute_y_true(x):
    """
    real function to generate y values
    Parameters:
        x: array of shape (n,), the input data
    returns:
        y: array of shape (n,), the output data
    """
    y = np.sin(x)+np.cos(2*x)
    return y

def generate_data_synthetic_periodic(N=1000, percent_outliers_size=0.35,x_min=-4, x_max=4,):
    """
    Generate synthetic data with a periodic pattern.
    Parameters:
        N: int, the number of data points to generate
        outliers_size: float percent the number of outliers to generate
        
    Returns:
        x: array of shape (N+outliers_size,), the input data
        y: array of shape (N+outlier_size,), the output data
    """
    np.random.seed(42)
    sample_outlier_size = int(N*percent_outliers_size)
    #sample_outlier_size=30
    #sample_to_generate = N-sample_outlier_size

    sample_to_generate = N-sample_outlier_size
    # Step 1: Generate a sequence of x values between x_min and x_max and add noise
    x = np.linspace(x_min,x_max, sample_to_generate) + np.random.normal(0, 0.1, sample_to_generate)

    # Step 2: Generate y values as a function of x with periodic characteristics
    y = compute_y_true(x)   
    
    
    # Step 3. Add some noise to the y values
    noise = np.random.normal(0, 0.5,sample_to_generate)  
    y += noise

    # Step 4: Add some outliers
    outiers_y = np.random.uniform(y.min(), y.max(), sample_outlier_size) 
    outiers_x = np.random.uniform(x.min(), x.max(), sample_outlier_size) 

    x = np.concatenate([x, outiers_x])
    y = np.concatenate([y, outiers_y])

    return x, y



def fourier_basis(x, num_terms):
    """Create a Fourier basis matrix.
    Parameters:
        x: array of shape (n,), the input data
        num_terms: int, the number of Fourier terms to use in the basis matrix
    returns:
        array of shape (n, 2*num_terms+1), the Fourier basis matrix
    """
    basis_matrix = np.ones((len(x), 1))  # Incluir término constante
    for j in range(1, num_terms + 1):
        basis_matrix = np.concatenate((basis_matrix, np.sin(j * x[:, np.newaxis]), np.cos(j * x[:, np.newaxis])), axis=1)
    return basis_matrix


