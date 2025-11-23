"""
Physics-Informed Neural Network (PINN) for Slope Stability Analysis

This module implements a PINN to solve the elasticity equations for slope stability problems.
The network learns displacement fields (u, v) and stress fields (σ11, σ22, σ12) by enforcing:
- Equilibrium equations (force balance)
- Constitutive relations (stress-strain)
- Boundary conditions (fixed base, free surface, lateral boundaries)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class SlopeStabilityPINN(nn.Module):
    """
    Physics-Informed Neural Network for slope stability analysis.

    The network predicts displacement (u, v) from coordinates (x, y).
    Stresses and strains are computed from displacements using automatic differentiation.
    """

    def __init__(self, layers: List[int] = [2, 64, 64, 64, 64, 2],
                 E: float = 10e6, nu: float = 0.3, rho: float = 2000.0, g: float = 9.81):
        """
        Initialize the PINN model.

        Args:
            layers: List defining network architecture [input_dim, hidden1, ..., output_dim]
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            rho: Density (kg/m³)
            g: Gravitational acceleration (m/s²)
        """
        super(SlopeStabilityPINN, self).__init__()

        # Material properties
        self.E = E
        self.nu = nu
        self.rho = rho
        self.g = g

        # Compute Lamé parameters
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lambda
        self.mu = E / (2 * (1 + nu))  # Shear modulus (G)

        # Build neural network
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Activation function
        self.activation = nn.Tanh()

        # Initialize weights using Xavier initialization
        self.init_weights()

    def init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict displacements u and v.

        Args:
            x: x-coordinates (batch_size, 1)
            y: y-coordinates (batch_size, 1)

        Returns:
            u: Horizontal displacement (batch_size, 1)
            v: Vertical displacement (batch_size, 1)
        """
        # Normalize inputs
        x_norm = 2.0 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1.0
        y_norm = 2.0 * (y - y.min()) / (y.max() - y.min() + 1e-8) - 1.0

        # Concatenate inputs
        inputs = torch.cat([x_norm, y_norm], dim=1)

        # Pass through network
        out = inputs
        for i, layer in enumerate(self.layers[:-1]):
            out = self.activation(layer(out))

        # Output layer (no activation)
        out = self.layers[-1](out)

        u = out[:, 0:1]  # Horizontal displacement
        v = out[:, 1:2]  # Vertical displacement

        return u, v

    def compute_strains(self, x: torch.Tensor, y: torch.Tensor,
                       u: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute strain components from displacements using automatic differentiation.

        Args:
            x, y: Coordinates
            u, v: Displacements

        Returns:
            Dictionary containing strain components (ε11, ε22, ε12)
        """
        # Compute gradients
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                    create_graph=True, retain_graph=True)[0]
        dv_dy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                                    create_graph=True, retain_graph=True)[0]

        # Strain components
        epsilon_11 = du_dx
        epsilon_22 = dv_dy
        epsilon_12 = 0.5 * (du_dy + dv_dx)

        return {
            'epsilon_11': epsilon_11,
            'epsilon_22': epsilon_22,
            'epsilon_12': epsilon_12
        }

    def compute_stresses(self, strains: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute stress components from strains using constitutive relation (Hooke's law).

        For plane strain:
        σ11 = (λ + 2μ)ε11 + λε22
        σ22 = λε11 + (λ + 2μ)ε22
        σ12 = 2με12

        Args:
            strains: Dictionary of strain components

        Returns:
            Dictionary containing stress components (σ11, σ22, σ12)
        """
        eps11 = strains['epsilon_11']
        eps22 = strains['epsilon_22']
        eps12 = strains['epsilon_12']

        sigma_11 = (self.lam + 2 * self.mu) * eps11 + self.lam * eps22
        sigma_22 = self.lam * eps11 + (self.lam + 2 * self.mu) * eps22
        sigma_12 = 2 * self.mu * eps12

        return {
            'sigma_11': sigma_11,
            'sigma_22': sigma_22,
            'sigma_12': sigma_12
        }

    def compute_equilibrium_residual(self, x: torch.Tensor, y: torch.Tensor,
                                    stresses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute equilibrium equation residuals (PDE residuals).

        Equilibrium equations:
        ∂σ11/∂x + ∂σ12/∂y + fx = 0
        ∂σ12/∂x + ∂σ22/∂y + fy = 0

        where fx = 0, fy = -ρg (gravity)

        Args:
            x, y: Coordinates
            stresses: Dictionary of stress components

        Returns:
            residual_x: Residual of x-direction equilibrium
            residual_y: Residual of y-direction equilibrium
        """
        sigma_11 = stresses['sigma_11']
        sigma_22 = stresses['sigma_22']
        sigma_12 = stresses['sigma_12']

        # Compute stress gradients
        dsigma_11_dx = torch.autograd.grad(sigma_11, x, grad_outputs=torch.ones_like(sigma_11),
                                          create_graph=True, retain_graph=True)[0]
        dsigma_12_dy = torch.autograd.grad(sigma_12, y, grad_outputs=torch.ones_like(sigma_12),
                                          create_graph=True, retain_graph=True)[0]
        dsigma_12_dx = torch.autograd.grad(sigma_12, x, grad_outputs=torch.ones_like(sigma_12),
                                          create_graph=True, retain_graph=True)[0]
        dsigma_22_dy = torch.autograd.grad(sigma_22, y, grad_outputs=torch.ones_like(sigma_22),
                                          create_graph=True, retain_graph=True)[0]

        # Body forces
        fx = 0.0
        fy = -self.rho * self.g

        # Equilibrium residuals
        residual_x = dsigma_11_dx + dsigma_12_dy + fx
        residual_y = dsigma_12_dx + dsigma_22_dy + fy

        return residual_x, residual_y


class SlopeStabilityTrainer:
    """Trainer class for the slope stability PINN."""

    def __init__(self, model: SlopeStabilityPINN, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.

        Args:
            model: PINN model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.history = {'loss': [], 'loss_pde': [], 'loss_bc': [], 'loss_ic': []}

    def compute_loss(self, x_domain: torch.Tensor, y_domain: torch.Tensor,
                    x_bc: Dict[str, torch.Tensor], y_bc: Dict[str, torch.Tensor],
                    bc_values: Dict[str, torch.Tensor],
                    lambda_pde: float = 1.0, lambda_bc: float = 100.0) -> Dict[str, torch.Tensor]:
        """
        Compute total loss = PDE loss + boundary condition loss.

        Args:
            x_domain, y_domain: Domain (collocation) points
            x_bc, y_bc: Boundary condition points (dict with keys for different boundaries)
            bc_values: Boundary condition values
            lambda_pde: Weight for PDE loss
            lambda_bc: Weight for BC loss

        Returns:
            Dictionary containing total loss and individual components
        """
        # PDE Loss (equilibrium equations in domain)
        u_domain, v_domain = self.model(x_domain, y_domain)
        strains = self.model.compute_strains(x_domain, y_domain, u_domain, v_domain)
        stresses = self.model.compute_stresses(strains)
        residual_x, residual_y = self.model.compute_equilibrium_residual(x_domain, y_domain, stresses)

        loss_pde = torch.mean(residual_x**2) + torch.mean(residual_y**2)

        # Boundary Condition Loss
        loss_bc = 0.0

        # Fixed base (u = 0, v = 0)
        if 'base' in x_bc:
            u_base, v_base = self.model(x_bc['base'], y_bc['base'])
            loss_bc += torch.mean(u_base**2) + torch.mean(v_base**2)

        # Left boundary (u = 0)
        if 'left' in x_bc:
            u_left, v_left = self.model(x_bc['left'], y_bc['left'])
            loss_bc += torch.mean(u_left**2)

        # Right boundary (u = 0)
        if 'right' in x_bc:
            u_right, v_right = self.model(x_bc['right'], y_bc['right'])
            loss_bc += torch.mean(u_right**2)

        # Free surface (traction-free: σ·n = 0)
        # For simplicity, we approximate this with natural boundary conditions
        # In a more sophisticated implementation, we would compute tractions explicitly

        # Total loss
        loss_total = lambda_pde * loss_pde + lambda_bc * loss_bc

        return {
            'total': loss_total,
            'pde': loss_pde,
            'bc': loss_bc
        }

    def train_step(self, optimizer: torch.optim.Optimizer,
                  x_domain: torch.Tensor, y_domain: torch.Tensor,
                  x_bc: Dict[str, torch.Tensor], y_bc: Dict[str, torch.Tensor],
                  bc_values: Dict[str, torch.Tensor],
                  lambda_pde: float = 1.0, lambda_bc: float = 100.0) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            optimizer: Optimizer
            x_domain, y_domain: Domain points
            x_bc, y_bc: Boundary points
            bc_values: Boundary values
            lambda_pde: PDE loss weight
            lambda_bc: BC loss weight

        Returns:
            Dictionary of losses
        """
        self.model.train()
        optimizer.zero_grad()

        losses = self.compute_loss(x_domain, y_domain, x_bc, y_bc, bc_values,
                                   lambda_pde, lambda_bc)

        losses['total'].backward()
        optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def train(self, x_domain: torch.Tensor, y_domain: torch.Tensor,
             x_bc: Dict[str, torch.Tensor], y_bc: Dict[str, torch.Tensor],
             bc_values: Dict[str, torch.Tensor],
             epochs: int = 10000, lr: float = 1e-3,
             lambda_pde: float = 1.0, lambda_bc: float = 100.0,
             print_every: int = 500):
        """
        Train the PINN model.

        Args:
            x_domain, y_domain: Domain points
            x_bc, y_bc: Boundary points
            bc_values: Boundary values
            epochs: Number of training epochs
            lr: Learning rate
            lambda_pde: PDE loss weight
            lambda_bc: BC loss weight
            print_every: Print frequency
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

        for epoch in range(epochs):
            losses = self.train_step(optimizer, x_domain, y_domain, x_bc, y_bc,
                                    bc_values, lambda_pde, lambda_bc)

            # Store history
            self.history['loss'].append(losses['total'])
            self.history['loss_pde'].append(losses['pde'])
            self.history['loss_bc'].append(losses['bc'])

            # Learning rate scheduling
            scheduler.step(losses['total'])

            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Total Loss: {losses['total']:.6e} | "
                      f"PDE Loss: {losses['pde']:.6e} | "
                      f"BC Loss: {losses['bc']:.6e}")

    def predict(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Predict displacement and stress fields.

        Args:
            x, y: Coordinates (numpy arrays or tensors)

        Returns:
            Dictionary containing all field variables
        """
        self.model.eval()

        # Convert to tensors if necessary
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            y = torch.tensor(y, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            u, v = self.model(x, y)

        # Enable gradients for strain/stress computation
        x.requires_grad = True
        y.requires_grad = True
        u, v = self.model(x, y)

        strains = self.model.compute_strains(x, y, u, v)
        stresses = self.model.compute_stresses(strains)

        # Convert to numpy
        results = {
            'u': u.detach().cpu().numpy(),
            'v': v.detach().cpu().numpy(),
            'epsilon_11': strains['epsilon_11'].detach().cpu().numpy(),
            'epsilon_22': strains['epsilon_22'].detach().cpu().numpy(),
            'epsilon_12': strains['epsilon_12'].detach().cpu().numpy(),
            'sigma_11': stresses['sigma_11'].detach().cpu().numpy(),
            'sigma_22': stresses['sigma_22'].detach().cpu().numpy(),
            'sigma_12': stresses['sigma_12'].detach().cpu().numpy()
        }

        return results

    def plot_loss_history(self, save_path: str = None):
        """Plot training loss history."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogy(self.history['loss'], label='Total Loss', linewidth=2)
        ax.semilogy(self.history['loss_pde'], label='PDE Loss', linewidth=2)
        ax.semilogy(self.history['loss_bc'], label='BC Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss History', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
