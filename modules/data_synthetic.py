import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data_synthetic_periodic(N=1000, percent_outliers_size=0.4,x_min=-4, x_max=4,):
    """
    Generate synthetic data with a periodic pattern.
    Parameters:
        N: int, the number of data points to generate
        percent_outliers_size: float, percent of the number of outliers to generate
        
    Returns:
        x: array of shape (N+outliers_size,), the input data
        y: array of shape (N+outlier_size,), the output data
    """
    sample_outlier_size = int(N*percent_outliers_size)
    sample_to_generate = N-sample_outlier_size

    np.random.seed(42)
    # Step 1: Generate a sequence of x values between x_min and x_max and add noise
    x = np.linspace(x_min,x_max, sample_to_generate) + np.random.normal(0, 0.1, sample_to_generate)

    # Step 2: Generate y values as a function of x with periodic characteristics
    y = np.sin(x)+np.cos(2*x)
    

    # Step 3. Add some noise to the y values
    noise = np.random.normal(0, 0.5,sample_to_generate)  
    y += noise

    # Step 4: Add some outliers
    outiers_y = np.random.uniform(y.min(), y.max(), sample_outliers_size)
    outiers_x = np.random.uniform(x.min(), x.max(), sample_outliers_size)
    x = np.concatenate([x, outiers_x])
    y = np.concatenate([y, outiers_y])

    return x, y

def plot_data(x, y,y_true):
    """
    Plot the data.
    Parameters:
        x: array of shape (N,), the input data
        y: array of shape (N,), the output data
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, label='Data Points', color='blue', alpha=0.5)
    sns.lineplot(x=x, y=y_true, label='True Function', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Synthetic Data with Periodic Pattern')
    plt.legend()
    plt.savefig('synthetic_data.png')
    plt.show()