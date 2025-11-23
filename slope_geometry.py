"""
Slope Geometry Definition and Sampling Utilities

This module provides functions to define slope geometry, generate collocation points,
and sample boundary conditions for PINN training.
"""

import numpy as np
import torch
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class SlopeGeometry:
    """Class to define and manage slope geometry."""

    def __init__(self, crest_point: Tuple[float, float] = (8.0, 10.0),
                 toe_point: Tuple[float, float] = (18.0, 5.0),
                 base_height: float = 0.0,
                 left_boundary: float = 0.0,
                 right_boundary: float = 20.0):
        """
        Initialize slope geometry.

        Args:
            crest_point: (x, y) coordinates of slope crest
            toe_point: (x, y) coordinates of slope toe
            base_height: y-coordinate of base (fixed boundary)
            left_boundary: x-coordinate of left boundary
            right_boundary: x-coordinate of right boundary
        """
        self.crest_point = np.array(crest_point)
        self.toe_point = np.array(toe_point)
        self.base_height = base_height
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        # Define slope surface equation
        # Linear slope from crest to toe
        self.slope_angle = np.arctan2(
            toe_point[1] - crest_point[1],
            toe_point[0] - crest_point[0]
        )

    def get_slope_surface_y(self, x: np.ndarray) -> np.ndarray:
        """
        Get y-coordinate of slope surface for given x-coordinates.

        Args:
            x: x-coordinates

        Returns:
            y: y-coordinates on slope surface
        """
        x = np.atleast_1d(x)
        y = np.zeros_like(x)

        # Left of crest: horizontal
        mask_left = x <= self.crest_point[0]
        y[mask_left] = self.crest_point[1]

        # Between crest and toe: sloped
        mask_slope = (x > self.crest_point[0]) & (x <= self.toe_point[0])
        slope = (self.toe_point[1] - self.crest_point[1]) / (self.toe_point[0] - self.crest_point[0])
        y[mask_slope] = self.crest_point[1] + slope * (x[mask_slope] - self.crest_point[0])

        # Right of toe: horizontal
        mask_right = x > self.toe_point[0]
        y[mask_right] = self.toe_point[1]

        return y

    def is_inside_domain(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Check if points (x, y) are inside the slope domain.

        Args:
            x, y: Coordinates

        Returns:
            mask: Boolean array indicating if points are inside
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        # Check if within x boundaries
        mask_x = (x >= self.left_boundary) & (x <= self.right_boundary)

        # Check if above base and below surface
        surface_y = self.get_slope_surface_y(x)
        mask_y = (y >= self.base_height) & (y <= surface_y)

        return mask_x & mask_y

    def sample_domain_points(self, n_points: int, method: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample collocation points within the domain.

        Args:
            n_points: Number of points to sample
            method: Sampling method ('uniform', 'latin_hypercube', 'adaptive')

        Returns:
            x, y: Coordinates of sampled points
        """
        if method == 'uniform':
            # Uniform random sampling with rejection
            x_samples = []
            y_samples = []

            x_range = self.right_boundary - self.left_boundary
            y_max = self.crest_point[1]
            y_range = y_max - self.base_height

            while len(x_samples) < n_points:
                # Generate candidate points
                n_candidates = n_points * 2
                x_cand = np.random.uniform(self.left_boundary, self.right_boundary, n_candidates)
                y_cand = np.random.uniform(self.base_height, y_max, n_candidates)

                # Filter points inside domain
                mask = self.is_inside_domain(x_cand, y_cand)
                x_samples.extend(x_cand[mask])
                y_samples.extend(y_cand[mask])

            x = np.array(x_samples[:n_points])
            y = np.array(y_samples[:n_points])

        elif method == 'latin_hypercube':
            # Latin Hypercube Sampling
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=2)
            samples = sampler.random(n=n_points * 2)

            # Scale to domain bounds
            x_cand = self.left_boundary + samples[:, 0] * (self.right_boundary - self.left_boundary)
            y_cand = self.base_height + samples[:, 1] * (self.crest_point[1] - self.base_height)

            # Filter points inside domain
            mask = self.is_inside_domain(x_cand, y_cand)
            x = x_cand[mask][:n_points]
            y = y_cand[mask][:n_points]

        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return x, y

    def sample_boundary_points(self, n_points_per_boundary: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Sample points on domain boundaries.

        Args:
            n_points_per_boundary: Number of points per boundary

        Returns:
            Dictionary with keys: 'base', 'left', 'right', 'surface'
            Each value is a tuple (x, y) of boundary points
        """
        boundaries = {}

        # Base boundary (fixed)
        x_base = np.linspace(self.left_boundary, self.right_boundary, n_points_per_boundary)
        y_base = np.full_like(x_base, self.base_height)
        boundaries['base'] = (x_base, y_base)

        # Left boundary
        x_left = np.full(n_points_per_boundary, self.left_boundary)
        y_left = np.linspace(self.base_height, self.crest_point[1], n_points_per_boundary)
        boundaries['left'] = (x_left, y_left)

        # Right boundary
        x_right = np.full(n_points_per_boundary, self.right_boundary)
        y_right = np.linspace(self.base_height, self.toe_point[1], n_points_per_boundary)
        boundaries['right'] = (x_right, y_right)

        # Surface boundary (free surface)
        x_surface = np.linspace(self.left_boundary, self.right_boundary, n_points_per_boundary * 2)
        y_surface = self.get_slope_surface_y(x_surface)
        boundaries['surface'] = (x_surface, y_surface)

        return boundaries

    def plot_geometry(self, show_points: bool = False, n_points: int = 1000,
                     save_path: str = None):
        """
        Plot the slope geometry.

        Args:
            show_points: Whether to show sampled collocation points
            n_points: Number of points to sample if show_points=True
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define polygon vertices for slope domain
        x_surface = np.linspace(self.left_boundary, self.right_boundary, 100)
        y_surface = self.get_slope_surface_y(x_surface)

        vertices = []
        # Base
        vertices.append([self.left_boundary, self.base_height])
        vertices.append([self.right_boundary, self.base_height])
        # Right boundary
        vertices.append([self.right_boundary, self.toe_point[1]])
        # Surface (reversed)
        for i in range(len(x_surface) - 1, -1, -1):
            vertices.append([x_surface[i], y_surface[i]])

        polygon = Polygon(vertices, facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.5)
        ax.add_patch(polygon)

        # Plot boundaries
        ax.plot([self.left_boundary, self.right_boundary], [self.base_height, self.base_height],
                'b-', linewidth=3, label='Base (fixed)')
        ax.plot([self.left_boundary, self.left_boundary], [self.base_height, self.crest_point[1]],
                'c-', linewidth=2, label='Left boundary')
        ax.plot([self.right_boundary, self.right_boundary], [self.base_height, self.toe_point[1]],
                'm-', linewidth=2, label='Right boundary')
        ax.plot(x_surface, y_surface, 'k-', linewidth=3, label='Slope surface')

        # Mark crest and toe
        ax.plot(self.crest_point[0], self.crest_point[1], 'ro', markersize=12, label='Crest')
        ax.plot(self.toe_point[0], self.toe_point[1], 'go', markersize=12, label='Toe')

        # Plot sampled points if requested
        if show_points:
            x_pts, y_pts = self.sample_domain_points(n_points)
            ax.scatter(x_pts, y_pts, c='blue', s=1, alpha=0.3, label=f'Collocation points (n={n_points})')

        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title('Slope Geometry', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def prepare_training_data(geometry: SlopeGeometry, n_domain: int = 5000,
                         n_boundary: int = 200, device: str = 'cpu') -> Dict:
    """
    Prepare training data (collocation points and boundary conditions) for PINN.

    Args:
        geometry: SlopeGeometry object
        n_domain: Number of domain (collocation) points
        n_boundary: Number of boundary points per boundary
        device: Device to place tensors on

    Returns:
        Dictionary containing training data as PyTorch tensors
    """
    # Sample domain points
    x_domain, y_domain = geometry.sample_domain_points(n_domain, method='uniform')

    # Sample boundary points
    boundaries = geometry.sample_boundary_points(n_boundary)

    # Convert to PyTorch tensors
    data = {
        'x_domain': torch.tensor(x_domain.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device),
        'y_domain': torch.tensor(y_domain.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device),
        'x_bc': {},
        'y_bc': {},
        'bc_values': {}
    }

    for boundary_name, (x_bc, y_bc) in boundaries.items():
        data['x_bc'][boundary_name] = torch.tensor(x_bc.reshape(-1, 1), dtype=torch.float32,
                                                    requires_grad=True).to(device)
        data['y_bc'][boundary_name] = torch.tensor(y_bc.reshape(-1, 1), dtype=torch.float32,
                                                    requires_grad=True).to(device)

    return data


def create_prediction_grid(geometry: SlopeGeometry, nx: int = 100, ny: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a regular grid for prediction and visualization.

    Args:
        geometry: SlopeGeometry object
        nx: Number of grid points in x-direction
        ny: Number of grid points in y-direction

    Returns:
        X, Y: Meshgrid arrays (flattened for prediction)
    """
    x = np.linspace(geometry.left_boundary, geometry.right_boundary, nx)
    y = np.linspace(geometry.base_height, geometry.crest_point[1], ny)
    X, Y = np.meshgrid(x, y)

    # Mask points outside domain
    mask = geometry.is_inside_domain(X.flatten(), Y.flatten())

    X_masked = X.flatten()[mask]
    Y_masked = Y.flatten()[mask]

    return X_masked, Y_masked, X.shape, mask


def plot_results(geometry: SlopeGeometry, predictions: Dict[str, np.ndarray],
                X_masked: np.ndarray, Y_masked: np.ndarray,
                grid_shape: Tuple[int, int], mask: np.ndarray,
                save_path: str = None):
    """
    Plot prediction results (displacements and stresses).

    Args:
        geometry: SlopeGeometry object
        predictions: Dictionary of predicted fields
        X_masked, Y_masked: Masked grid coordinates
        grid_shape: Shape of original grid
        mask: Boolean mask for valid points
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    fields = [
        ('u', 'Horizontal Displacement u (m)'),
        ('v', 'Vertical Displacement v (m)'),
        ('sigma_11', 'Stress σ₁₁ (MPa)'),
        ('sigma_22', 'Stress σ₂₂ (MPa)'),
        ('sigma_12', 'Stress σ₁₂ (MPa)'),
    ]

    # Create full grids with NaN for masked regions
    def create_grid(values):
        grid = np.full(grid_shape[0] * grid_shape[1], np.nan)
        grid[mask] = values.flatten()
        return grid.reshape(grid_shape)

    for idx, (field, title) in enumerate(fields):
        ax = axes[idx // 3, idx % 3]

        if field in predictions:
            values = predictions[field]
            if field.startswith('sigma'):
                values = values / 1e6  # Convert to MPa

            grid = create_grid(values)

            # Create meshgrid for plotting
            x = np.linspace(geometry.left_boundary, geometry.right_boundary, grid_shape[1])
            y = np.linspace(geometry.base_height, geometry.crest_point[1], grid_shape[0])
            X, Y = np.meshgrid(x, y)

            contour = ax.contourf(X, Y, grid, levels=20, cmap='RdBu_r')
            plt.colorbar(contour, ax=ax)

            ax.set_xlabel('x (m)', fontsize=10)
            ax.set_ylabel('y (m)', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')

    # Displacement magnitude
    ax = axes[1, 2]
    u_mag = np.sqrt(predictions['u']**2 + predictions['v']**2)
    grid = create_grid(u_mag)
    x = np.linspace(geometry.left_boundary, geometry.right_boundary, grid_shape[1])
    y = np.linspace(geometry.base_height, geometry.crest_point[1], grid_shape[0])
    X, Y = np.meshgrid(x, y)
    contour = ax.contourf(X, Y, grid, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.set_title('Displacement Magnitude (m)', fontsize=11, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
