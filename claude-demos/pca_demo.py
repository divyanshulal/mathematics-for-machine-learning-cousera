#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) Demonstrations
Practical examples of concepts from Course 3: PCA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, make_blobs


def demonstrate_pca_basics():
    """Basic PCA on 2D data"""
    print("=" * 60)
    print("1. PCA BASICS - 2D DATA")
    print("=" * 60)

    # Generate correlated 2D data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[3, 1.5], [1.5, 1]]
    data = np.random.multivariate_normal(mean, cov, 300)

    print(f"\nData shape: {data.shape}")
    print(f"Data mean: {np.mean(data, axis=0)}")

    # Center the data
    data_centered = data - np.mean(data, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    print(f"\nCovariance matrix:\n{cov_matrix}")

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nEigenvectors (Principal Components):\n{eigenvectors}")

    # Explained variance
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    print(f"\nExplained variance ratio: {explained_variance_ratio}")

    # Visualize
    plt.figure(figsize=(12, 5))

    # Original data with principal components
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=30)

    # Plot principal components
    origin = np.mean(data, axis=0)
    for i in range(2):
        pc = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 3
        plt.arrow(origin[0], origin[1], pc[0], pc[1],
                 head_width=0.3, head_length=0.3, fc=f'C{i+1}', ec=f'C{i+1}',
                 linewidth=3, label=f'PC{i+1} ({explained_variance_ratio[i]:.1%})')

    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Original Data with Principal Components')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Transformed data (in PC space)
    data_transformed = data_centered @ eigenvectors
    plt.subplot(1, 2, 2)
    plt.scatter(data_transformed[:, 0], data_transformed[:, 1], alpha=0.6, s=30)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Data in Principal Component Space')
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('claude-demos/pca_basics.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'pca_basics.png'")


def demonstrate_dimensionality_reduction():
    """PCA for dimensionality reduction on Iris dataset"""
    print("\n" + "=" * 60)
    print("2. DIMENSIONALITY REDUCTION - IRIS DATASET")
    print("=" * 60)

    # Load Iris dataset (4D data)
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    print(f"\nOriginal data shape: {X.shape} (4 features)")
    print(f"Features: {iris.feature_names}")

    # Apply PCA
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)

    print(f"\nExplained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nCumulative explained variance:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"  PC1-PC{i+1}: {cum_var:.4f} ({cum_var*100:.2f}%)")

    # Visualize
    fig = plt.figure(figsize=(14, 5))

    # Scree plot
    ax1 = plt.subplot(1, 3, 1)
    plt.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7)
    plt.plot(range(1, 5), cumulative_variance, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.xticks(range(1, 5))
    plt.grid(True, alpha=0.3)

    # 2D projection (PC1 vs PC2)
    ax2 = plt.subplot(1, 3, 2)
    colors = ['red', 'green', 'blue']
    for i, target_name in enumerate(target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                   alpha=0.8, color=colors[i], label=target_name, s=50)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('Iris Dataset - 2D PCA Projection')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2D projection (PC2 vs PC3)
    ax3 = plt.subplot(1, 3, 3)
    for i, target_name in enumerate(target_names):
        plt.scatter(X_pca[y == i, 1], X_pca[y == i, 2],
                   alpha=0.8, color=colors[i], label=target_name, s=50)
    plt.xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    plt.title('Iris Dataset - Alternative View')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('claude-demos/pca_iris.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'pca_iris.png'")


def demonstrate_reconstruction():
    """PCA reconstruction with different numbers of components"""
    print("\n" + "=" * 60)
    print("3. DATA RECONSTRUCTION WITH PCA")
    print("=" * 60)

    # Generate 2D data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[4, 1.8], [1.8, 1]]
    original_data = np.random.multivariate_normal(mean, cov, 200)

    # Sample point to reconstruct
    sample_idx = 0
    sample_point = original_data[sample_idx]

    print(f"\nOriginal sample point: {sample_point}")

    # PCA with different numbers of components
    n_components_list = [1, 2]

    plt.figure(figsize=(12, 5))

    for idx, n_components in enumerate(n_components_list):
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(original_data)
        reconstructed = pca.inverse_transform(transformed)

        sample_reconstructed = reconstructed[sample_idx]
        reconstruction_error = np.linalg.norm(sample_point - sample_reconstructed)

        print(f"\nWith {n_components} component(s):")
        print(f"  Reconstructed point: {sample_reconstructed}")
        print(f"  Reconstruction error: {reconstruction_error:.4f}")
        print(f"  Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

        plt.subplot(1, 2, idx + 1)
        plt.scatter(original_data[:, 0], original_data[:, 1],
                   alpha=0.4, s=30, label='Original data')
        plt.scatter(reconstructed[:, 0], reconstructed[:, 1],
                   alpha=0.4, s=30, label='Reconstructed data')

        # Highlight sample point
        plt.scatter(sample_point[0], sample_point[1],
                   color='red', s=200, marker='*',
                   edgecolors='black', linewidths=2, label='Original sample')
        plt.scatter(sample_reconstructed[0], sample_reconstructed[1],
                   color='orange', s=200, marker='*',
                   edgecolors='black', linewidths=2, label='Reconstructed sample')

        # Draw line between original and reconstructed
        plt.plot([sample_point[0], sample_reconstructed[0]],
                [sample_point[1], sample_reconstructed[1]],
                'k--', linewidth=2, alpha=0.5)

        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title(f'Reconstruction with {n_components} PC(s)\n'
                 f'Error: {reconstruction_error:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

    plt.tight_layout()
    plt.savefig('claude-demos/pca_reconstruction.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'pca_reconstruction.png'")


def demonstrate_data_whitening():
    """Demonstrate PCA whitening"""
    print("\n" + "=" * 60)
    print("4. PCA WHITENING")
    print("=" * 60)

    # Generate correlated data
    np.random.seed(42)
    mean = [0, 0]
    cov = [[3, 1.5], [1.5, 1]]
    data = np.random.multivariate_normal(mean, cov, 300)

    # Apply PCA with whitening
    pca = PCA(whiten=True)
    data_whitened = pca.fit_transform(data)

    print(f"\nOriginal data:")
    print(f"  Mean: {np.mean(data, axis=0)}")
    print(f"  Std: {np.std(data, axis=0)}")
    print(f"  Covariance:\n{np.cov(data.T)}")

    print(f"\nWhitened data:")
    print(f"  Mean: {np.mean(data_whitened, axis=0)}")
    print(f"  Std: {np.std(data_whitened, axis=0)}")
    print(f"  Covariance:\n{np.cov(data_whitened.T)}")

    # Visualize
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, s=30)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Original Data (Correlated)')
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.scatter(data_whitened[:, 0], data_whitened[:, 1], alpha=0.6, s=30)
    plt.xlabel('PC1 (whitened)')
    plt.ylabel('PC2 (whitened)')
    plt.title('Whitened Data (Uncorrelated, Unit Variance)')
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('claude-demos/pca_whitening.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'pca_whitening.png'")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS DEMONSTRATIONS")
    print("Mathematics for Machine Learning - Course 3")
    print("=" * 60)

    demonstrate_pca_basics()
    demonstrate_dimensionality_reduction()
    demonstrate_reconstruction()
    demonstrate_data_whitening()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
