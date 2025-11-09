#!/usr/bin/env python3
"""
Welcome Demo for Mathematics for Machine Learning
Imperial College London - Coursera Specialization

This demo showcases key mathematical concepts from the three courses:
1. Linear Algebra
2. Multivariate Calculus
3. Principal Component Analysis (PCA)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def print_welcome():
    """Display welcome message"""
    print("=" * 70)
    print(" " * 15 + "WELCOME TO")
    print(" " * 5 + "MATHEMATICS FOR MACHINE LEARNING")
    print(" " * 10 + "Imperial College London - Coursera")
    print("=" * 70)
    print("\nThis repository contains solutions and notes for:")
    print("  Course 1: Linear Algebra")
    print("  Course 2: Multivariate Calculus")
    print("  Course 3: Principal Component Analysis (PCA)")
    print("\n" + "=" * 70 + "\n")

def demo_linear_algebra():
    """Demonstrate basic linear algebra concepts"""
    print("📐 DEMO 1: Linear Algebra Basics")
    print("-" * 70)

    # Vector operations
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Dot Product: {np.dot(v1, v2)}")
    print(f"Cross Product: {np.cross(v1, v2)}")
    print(f"Magnitude of v1: {np.linalg.norm(v1):.2f}")

    # Matrix operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print(f"\nMatrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Matrix Multiplication A @ B:\n{A @ B}")
    print(f"Determinant of A: {np.linalg.det(A):.2f}")
    print(f"Inverse of A:\n{np.linalg.inv(A)}")

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    print()

def demo_calculus():
    """Demonstrate multivariate calculus concepts"""
    print("📊 DEMO 2: Multivariate Calculus")
    print("-" * 70)

    # Gradient descent example
    def f(x):
        """Simple quadratic function f(x) = x^2"""
        return x ** 2

    def gradient(x):
        """Gradient of f(x) = x^2 is 2x"""
        return 2 * x

    # Perform gradient descent
    x = 10.0  # Starting point
    learning_rate = 0.1
    iterations = 5

    print("Gradient Descent on f(x) = x²")
    print(f"Starting point: x = {x}")
    print(f"Learning rate: {learning_rate}")
    print()

    for i in range(iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}, gradient = {grad:.4f}")

    print(f"\nMinimum found at x ≈ {x:.4f}")

    # Partial derivatives example
    print("\n\nPartial Derivatives Example:")
    print("For f(x, y) = x² + y²")
    x, y = 3, 4
    print(f"At point ({x}, {y}):")
    print(f"∂f/∂x = 2x = {2*x}")
    print(f"∂f/∂y = 2y = {2*y}")
    print(f"Gradient vector: [{2*x}, {2*y}]")
    print()

def demo_pca():
    """Demonstrate PCA concept"""
    print("🔍 DEMO 3: Principal Component Analysis")
    print("-" * 70)

    # Generate sample data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[3, 1.5], [1.5, 1]]
    data = np.random.multivariate_normal(mean, cov, 200)

    print("Performing PCA on 2D dataset...")
    print(f"Data shape: {data.shape}")
    print(f"Mean: {data.mean(axis=0)}")

    # Center the data
    data_centered = data - data.mean(axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    print(f"\nCovariance Matrix:\n{cov_matrix}")

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\nEigenvalues (variance explained): {eigenvalues}")
    print(f"Principal Components (eigenvectors):\n{eigenvectors}")

    # Variance explained
    variance_explained = eigenvalues / eigenvalues.sum()
    print(f"\nVariance explained by PC1: {variance_explained[0]:.2%}")
    print(f"Variance explained by PC2: {variance_explained[1]:.2%}")

    # Project onto first principal component
    pc1 = eigenvectors[:, 0]
    projection = data_centered @ pc1
    print(f"\nFirst 5 projections onto PC1: {projection[:5]}")
    print()

def demo_ml_application():
    """Show how these concepts apply to machine learning"""
    print("🤖 DEMO 4: Application to Machine Learning")
    print("-" * 70)

    print("How these mathematical concepts are used in ML:")
    print()
    print("LINEAR ALGEBRA:")
    print("  • Neural networks use matrix multiplication for forward propagation")
    print("  • Feature vectors represent data points")
    print("  • Weight matrices transform inputs to outputs")
    print()
    print("MULTIVARIATE CALCULUS:")
    print("  • Gradient descent optimizes neural network weights")
    print("  • Backpropagation computes gradients using chain rule")
    print("  • Loss functions are minimized using derivatives")
    print()
    print("PCA:")
    print("  • Dimensionality reduction for high-dimensional data")
    print("  • Feature extraction and data compression")
    print("  • Noise reduction and visualization")
    print()

def main():
    """Run all demos"""
    print_welcome()

    try:
        demo_linear_algebra()
        input("Press Enter to continue to Calculus demo...")
        print("\n")

        demo_calculus()
        input("Press Enter to continue to PCA demo...")
        print("\n")

        demo_pca()
        input("Press Enter to see ML applications...")
        print("\n")

        demo_ml_application()

        print("=" * 70)
        print("\n✨ Thank you for exploring Mathematics for Machine Learning! ✨")
        print("\nFor more details, check out the course folders:")
        print("  • course1 - linear algebra/")
        print("  • course2 - multivariate calculus/")
        print("  • course3 - principle component analysis/")
        print("\n" + "=" * 70)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Thank you for your time!")

if __name__ == "__main__":
    main()
