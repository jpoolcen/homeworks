import numpy as np
import matplotlib.pyplot as plt
def fx(x1, x2, x3):
    """
    compute the function f(x1, x2, x3) = 10*x1**2 - x1*x2 - 5*x1*x3 + 5*x1 + 10*x2**2 - 11*x2*x3 - 2*x2 - 5*x3*x1 - 11*x3*x2 - 4*x2 + 6*x3 + 9
    parameters:
        x1 (array): float
        x2 (array): float
        x3 (array): float

    returns:array
    """
    return (10*x1**2 - x1*x2 - 10*x1*x3 + 5*x1 + 10*x2**2 - 22*x2*x3 - 2*x2 + 6*x3 + 9)

def grad_fx(x1, x2, x3):
    df_dx1 = 20*x1 - x2 - 10*x3 + 5
    df_dx2 = -x1 + 20*x2 - 22*x3 - 2
    df_dx3 = -10*x1 - 22*x2 + 6
    return np.array([df_dx1, df_dx2, df_dx3])


def plot_function_and_gradient(x1_range, x2_range, x3_fixed, num_points=20):
    x1 = np.linspace(x1_range[0], x1_range[1], num_points)
    x2 = np.linspace(x2_range[0], x2_range[1], num_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Calcular los valores de la función y el gradiente en el plano x3 = x3_fixed
    F = np.zeros_like(X1)
    G1 = np.zeros_like(X1)
    G2 = np.zeros_like(X2)
    
    for i in range(num_points):
        for j in range(num_points):
            F[i, j] = fx(X1[i, j], X2[i, j], x3_fixed)
            grad = grad_fx(X1[i, j], X2[i, j], x3_fixed)
            G1[i, j] = grad[0]
            G2[i, j] = grad[1]
    
    # Crear la figura
    fig = plt.figure(figsize=(14, 7))
    
    # Visualizar la función
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f'Función f(x1, x2, x3_fixed={x3_fixed})')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x1, x2, x3)')
    ax1.plot_surface(X1, X2, F, cmap='viridis', edgecolor='none', alpha=0.7)
    ax1.scatter(X1, X2, F, c=F, cmap='viridis', s=1)
    
    # Visualizar el gradiente
    ax2 = fig.add_subplot(122)
    ax2.set_title(f'Gradiente de f en x3_fixed={x3_fixed}')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    contour = ax2.contour(X1, X2,F, 20, cmap='viridis')
    plt.colorbar(contour)
    ax2.quiver(X1, X2, -G1, -G2, color='red')
    
    plt.tight_layout()
    plt.show()