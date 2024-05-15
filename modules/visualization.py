import matplotlib.pyplot as plt

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