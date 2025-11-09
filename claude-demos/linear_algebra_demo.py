#!/usr/bin/env python3
"""
Linear Algebra Demonstrations
Practical examples of concepts from Course 1: Linear Algebra
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_matrix_operations():
    """Basic matrix operations and transformations"""
    print("=" * 60)
    print("1. MATRIX OPERATIONS")
    print("=" * 60)

    # Define matrices
    A = np.array([[2, 1], [1, 3]])
    B = np.array([[1, 0], [0, 2]])

    print(f"\nMatrix A:\n{A}")
    print(f"\nMatrix B:\n{B}")

    # Matrix multiplication
    print(f"\nA × B =\n{A @ B}")

    # Matrix inverse
    A_inv = np.linalg.inv(A)
    print(f"\nA⁻¹ =\n{A_inv}")
    print(f"\nVerification A × A⁻¹ =\n{A @ A_inv}")

    # Determinant
    det_A = np.linalg.det(A)
    print(f"\ndet(A) = {det_A:.4f}")


def demonstrate_eigenvalues():
    """Eigenvalues and eigenvectors demonstration"""
    print("\n" + "=" * 60)
    print("2. EIGENVALUES AND EIGENVECTORS")
    print("=" * 60)

    # Create a symmetric matrix
    A = np.array([[4, 2], [2, 3]])

    print(f"\nMatrix A:\n{A}")

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nEigenvectors:\n{eigenvectors}")

    # Verify: A × v = λ × v
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lambda_val = eigenvalues[i]

        left_side = A @ v
        right_side = lambda_val * v

        print(f"\nEigenvector {i+1}:")
        print(f"A × v = {left_side}")
        print(f"λ × v = {right_side}")
        print(f"Verification: {np.allclose(left_side, right_side)}")


def demonstrate_linear_transformation():
    """Visualize linear transformations"""
    print("\n" + "=" * 60)
    print("3. LINEAR TRANSFORMATIONS")
    print("=" * 60)

    # Original unit square
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])

    # Transformation matrix (rotation + scaling)
    theta = np.pi / 4  # 45 degrees
    scale = 1.5
    T = scale * np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])

    # Apply transformation
    transformed_square = T @ square

    print(f"\nTransformation Matrix:\n{T}")

    # Visualize
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(square[0, :], square[1, :], 'b-', linewidth=2, label='Original')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('Original Unit Square')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(square[0, :], square[1, :], 'b--', linewidth=1, alpha=0.3, label='Original')
    plt.plot(transformed_square[0, :], transformed_square[1, :], 'r-', linewidth=2, label='Transformed')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('Linear Transformation (Rotation + Scale)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.savefig('claude-demos/linear_transformation.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'linear_transformation.png'")


def demonstrate_basis_vectors():
    """Demonstrate basis vectors and coordinate systems"""
    print("\n" + "=" * 60)
    print("4. BASIS VECTORS AND CHANGE OF BASIS")
    print("=" * 60)

    # Standard basis
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])

    # New basis
    b1 = np.array([2, 1])
    b2 = np.array([1, 2])

    # Vector in standard basis
    v_standard = np.array([3, 4])

    print(f"Vector in standard basis: {v_standard}")

    # Change of basis matrix
    B = np.column_stack([b1, b2])
    B_inv = np.linalg.inv(B)

    # Convert to new basis
    v_new = B_inv @ v_standard

    print(f"\nNew basis vectors:")
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    print(f"\nVector in new basis: {v_new}")

    # Verify
    v_reconstructed = v_new[0] * b1 + v_new[1] * b2
    print(f"\nReconstructed vector: {v_reconstructed}")
    print(f"Verification: {np.allclose(v_standard, v_reconstructed)}")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("LINEAR ALGEBRA DEMONSTRATIONS")
    print("Mathematics for Machine Learning - Course 1")
    print("=" * 60)

    demonstrate_matrix_operations()
    demonstrate_eigenvalues()
    demonstrate_linear_transformation()
    demonstrate_basis_vectors()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
