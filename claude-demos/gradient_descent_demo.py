#!/usr/bin/env python3
"""
Gradient Descent Demonstrations
Practical examples of concepts from Course 2: Multivariate Calculus
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def demonstrate_gradient_descent_1d():
    """Simple 1D gradient descent on a quadratic function"""
    print("=" * 60)
    print("1. GRADIENT DESCENT - 1D EXAMPLE")
    print("=" * 60)

    # Function: f(x) = (x - 3)^2 + 5
    def f(x):
        return (x - 3)**2 + 5

    # Derivative: f'(x) = 2(x - 3)
    def df(x):
        return 2 * (x - 3)

    # Gradient descent parameters
    x = 0.0  # Starting point
    learning_rate = 0.1
    iterations = 20

    # Store history
    x_history = [x]
    f_history = [f(x)]

    print(f"\nStarting point: x = {x:.4f}, f(x) = {f(x):.4f}")

    # Perform gradient descent
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        x_history.append(x)
        f_history.append(f(x))

        if i < 5 or i >= iterations - 2:
            print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}, gradient = {gradient:.4f}")
        elif i == 5:
            print("...")

    # Visualize
    x_range = np.linspace(-1, 7, 200)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_range, f(x_range), 'b-', linewidth=2, label='f(x)')
    plt.plot(x_history, f_history, 'ro-', markersize=4, label='Gradient Descent Path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent on f(x) = (x-3)² + 5')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(f_history, 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('Convergence Plot')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('claude-demos/gradient_descent_1d.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'gradient_descent_1d.png'")


def demonstrate_gradient_descent_2d():
    """2D gradient descent on a paraboloid"""
    print("\n" + "=" * 60)
    print("2. GRADIENT DESCENT - 2D EXAMPLE")
    print("=" * 60)

    # Function: f(x, y) = x^2 + y^2 (simple paraboloid)
    def f(x, y):
        return x**2 + y**2

    # Gradient: ∇f = [2x, 2y]
    def grad_f(x, y):
        return np.array([2*x, 2*y])

    # Gradient descent parameters
    pos = np.array([4.0, 3.0])  # Starting point
    learning_rate = 0.1
    iterations = 30

    # Store history
    path = [pos.copy()]

    print(f"\nStarting point: ({pos[0]:.4f}, {pos[1]:.4f}), f = {f(*pos):.4f}")

    # Perform gradient descent
    for i in range(iterations):
        gradient = grad_f(*pos)
        pos = pos - learning_rate * gradient
        path.append(pos.copy())

        if i < 3 or i >= iterations - 2:
            print(f"Iteration {i+1}: ({pos[0]:.4f}, {pos[1]:.4f}), f = {f(*pos):.4f}")
        elif i == 3:
            print("...")

    path = np.array(path)

    # Visualize
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    ax1.plot(path[:, 0], path[:, 1], [f(*p) for p in path], 'ro-', markersize=4, linewidth=2, label='Gradient Descent')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x, y)')
    ax1.set_title('3D View: Gradient Descent on f(x,y) = x² + y²')
    ax1.legend()

    # Contour plot
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(path[:, 0], path[:, 1], 'ro-', markersize=6, linewidth=2, label='Gradient Descent Path')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour View')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('claude-demos/gradient_descent_2d.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'gradient_descent_2d.png'")


def demonstrate_learning_rate_comparison():
    """Compare different learning rates"""
    print("\n" + "=" * 60)
    print("3. LEARNING RATE COMPARISON")
    print("=" * 60)

    def f(x):
        return x**2

    def df(x):
        return 2*x

    learning_rates = [0.01, 0.1, 0.5, 0.9]
    iterations = 20
    starting_point = 5.0

    plt.figure(figsize=(12, 8))

    for idx, lr in enumerate(learning_rates):
        x = starting_point
        history = [x]

        for _ in range(iterations):
            x = x - lr * df(x)
            history.append(x)

        plt.subplot(2, 2, idx + 1)
        plt.plot(history, 'o-', linewidth=2, markersize=6)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Minimum')
        plt.xlabel('Iteration')
        plt.ylabel('x')
        plt.title(f'Learning Rate = {lr}')
        plt.grid(True)
        plt.legend()

        print(f"\nLearning Rate {lr}:")
        print(f"  Final x: {history[-1]:.6f}")
        print(f"  Final f(x): {f(history[-1]):.6f}")
        if abs(history[-1]) > 100:
            print(f"  Status: DIVERGED!")
        elif abs(history[-1]) < 0.001:
            print(f"  Status: Converged well")
        else:
            print(f"  Status: Converging...")

    plt.tight_layout()
    plt.savefig('claude-demos/learning_rate_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'learning_rate_comparison.png'")


def demonstrate_jacobian_hessian():
    """Demonstrate Jacobian and Hessian matrices"""
    print("\n" + "=" * 60)
    print("4. JACOBIAN AND HESSIAN MATRICES")
    print("=" * 60)

    # Function f(x, y) = [x^2 + y^2, x*y]
    def f(x, y):
        return np.array([x**2 + y**2, x*y])

    # Jacobian matrix
    def jacobian(x, y):
        return np.array([[2*x, 2*y],
                        [y, x]])

    # For g(x, y) = x^2 + y^2
    # Hessian matrix
    def hessian(x, y):
        return np.array([[2, 0],
                        [0, 2]])

    point = (3.0, 2.0)

    print(f"\nAt point ({point[0]}, {point[1]}):")
    print(f"\nFunction value f(x,y) = [x² + y², xy]:")
    print(f"  f = {f(*point)}")

    print(f"\nJacobian matrix:")
    J = jacobian(*point)
    print(J)

    print(f"\nHessian matrix (for g(x,y) = x² + y²):")
    H = hessian(*point)
    print(H)

    # Check if Hessian is positive definite (local minimum)
    eigenvalues = np.linalg.eigvals(H)
    print(f"\nHessian eigenvalues: {eigenvalues}")
    if np.all(eigenvalues > 0):
        print("All eigenvalues positive → Hessian is positive definite → Local minimum")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("GRADIENT DESCENT DEMONSTRATIONS")
    print("Mathematics for Machine Learning - Course 2")
    print("=" * 60)

    demonstrate_gradient_descent_1d()
    demonstrate_gradient_descent_2d()
    demonstrate_learning_rate_comparison()
    demonstrate_jacobian_hessian()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
