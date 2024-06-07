import matplotlib.pyplot as plt
import seaborn as sns

def plot_points(point, transformed_point):
    """
    Plot the original and transformed points

    parameters:
    -point: np.array, original point
    -transformed_point: np.array, transformed point
    
    """
    plt.figure(figsize=(8, 6))
    plt.plot(point[0], point[1], 'bo', label='Original Point')
    plt.plot(transformed_point[0][0], transformed_point[1][0], 'ro', label='Transformed Point')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Affine Transformation')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def show_image(original_image, transformed_image):
    """
    Show a image using matplotlib
    parameters:
    -original_image: np.array, image to show
    -transformed_image: np.array, transformed image to show
    """
    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2)

    # Show the original image in the first subplot
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')

    # Show the transformed image in the second subplot
    axes[1].imshow(transformed_image, cmap='gray')
    axes[1].set_title('Transformed Image')

    # Display the plot
    plt.show()

def plot_norm_excersice(X,Y,Z):
    """
    Plot the quadratic form sqrt(x.T  Sigma  x)
    
    parameters:
        -X: np.array, x values
        -Y: np.array, y values
        -Z: np.array, z values of the norm.
    """
    plt.figure(figsize=(6,6))
    plt.contour(X, Y, Z, levels=[1], colors='blue')
    plt.title(r'Norm  $\sqrt{x^T \Sigma x}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_periodic_data(x, y,y_true):
    """
    Plot the data.
    Parameters:
        x: array of shape (N,), the input data
        y: array of shape (N,), the output data
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, label='Data Points', color='blue', alpha=0.2)
    sns.lineplot(x=x, y=y_true, label='True Function', color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Synthetic Data with Periodic Pattern')
    plt.legend()
    plt.savefig('synthetic_data.png')
    plt.show()