"""
Training Script for Slope Stability PINN

This script demonstrates how to train a Physics-Informed Neural Network
for slope stability analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from slope_stability_pinn import SlopeStabilityPINN, SlopeStabilityTrainer
from slope_geometry import (SlopeGeometry, prepare_training_data,
                            create_prediction_grid, plot_results)
import os
from datetime import datetime


def main():
    """Main training function."""

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_slope_pinn_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # ========================
    # 1. Define Geometry
    # ========================
    print("\n" + "="*60)
    print("STEP 1: Defining Slope Geometry")
    print("="*60)

    geometry = SlopeGeometry(
        crest_point=(8.0, 10.0),     # Crest at (8m, 10m)
        toe_point=(18.0, 5.0),        # Toe at (18m, 5m)
        base_height=0.0,              # Base at y=0
        left_boundary=0.0,            # Left boundary at x=0
        right_boundary=20.0           # Right boundary at x=20m
    )

    # Plot and save geometry
    print("Plotting geometry...")
    geometry.plot_geometry(show_points=True, n_points=2000,
                          save_path=os.path.join(output_dir, 'geometry.png'))

    # ========================
    # 2. Material Properties
    # ========================
    print("\n" + "="*60)
    print("STEP 2: Setting Material Properties")
    print("="*60)

    # Soil properties (example values)
    E = 10e6        # Young's modulus: 10 MPa
    nu = 0.3        # Poisson's ratio
    rho = 2000.0    # Density: 2000 kg/m³
    g = 9.81        # Gravity: 9.81 m/s²

    print(f"Young's Modulus (E): {E/1e6:.1f} MPa")
    print(f"Poisson's Ratio (ν): {nu}")
    print(f"Density (ρ): {rho} kg/m³")
    print(f"Gravity (g): {g} m/s²")

    # ========================
    # 3. Initialize PINN Model
    # ========================
    print("\n" + "="*60)
    print("STEP 3: Initializing PINN Model")
    print("="*60)

    # Network architecture
    layers = [2, 64, 64, 64, 64, 2]  # [input_dim, hidden layers..., output_dim]
    print(f"Network architecture: {layers}")

    model = SlopeStabilityPINN(
        layers=layers,
        E=E,
        nu=nu,
        rho=rho,
        g=g
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # ========================
    # 4. Prepare Training Data
    # ========================
    print("\n" + "="*60)
    print("STEP 4: Preparing Training Data")
    print("="*60)

    n_domain = 5000      # Number of collocation points in domain
    n_boundary = 200     # Number of points per boundary

    print(f"Number of domain points: {n_domain}")
    print(f"Number of boundary points per boundary: {n_boundary}")

    training_data = prepare_training_data(
        geometry=geometry,
        n_domain=n_domain,
        n_boundary=n_boundary,
        device=device
    )

    print(f"Domain points shape: {training_data['x_domain'].shape}")
    print(f"Boundary conditions: {list(training_data['x_bc'].keys())}")

    # ========================
    # 5. Train the Model
    # ========================
    print("\n" + "="*60)
    print("STEP 5: Training PINN Model")
    print("="*60)

    trainer = SlopeStabilityTrainer(model=model, device=device)

    # Training hyperparameters
    epochs = 10000
    lr = 1e-3
    lambda_pde = 1.0
    lambda_bc = 100.0

    print(f"Training epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"PDE loss weight: {lambda_pde}")
    print(f"BC loss weight: {lambda_bc}")
    print("\nStarting training...")
    print("-" * 60)

    trainer.train(
        x_domain=training_data['x_domain'],
        y_domain=training_data['y_domain'],
        x_bc=training_data['x_bc'],
        y_bc=training_data['y_bc'],
        bc_values=training_data['bc_values'],
        epochs=epochs,
        lr=lr,
        lambda_pde=lambda_pde,
        lambda_bc=lambda_bc,
        print_every=1000
    )

    print("-" * 60)
    print("Training completed!")

    # Plot loss history
    print("\nPlotting loss history...")
    trainer.plot_loss_history(save_path=os.path.join(output_dir, 'loss_history.png'))

    # ========================
    # 6. Make Predictions
    # ========================
    print("\n" + "="*60)
    print("STEP 6: Making Predictions on Grid")
    print("="*60)

    # Create prediction grid
    X_masked, Y_masked, grid_shape, mask = create_prediction_grid(
        geometry=geometry,
        nx=100,
        ny=50
    )

    print(f"Prediction grid: {grid_shape}")
    print(f"Valid points: {len(X_masked)}")

    # Convert to tensors
    X_tensor = torch.tensor(X_masked.reshape(-1, 1), dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y_masked.reshape(-1, 1), dtype=torch.float32).to(device)

    # Predict
    print("Computing predictions...")
    predictions = trainer.predict(X_tensor, Y_tensor)

    # ========================
    # 7. Visualize Results
    # ========================
    print("\n" + "="*60)
    print("STEP 7: Visualizing Results")
    print("="*60)

    print("Plotting displacement and stress fields...")
    plot_results(
        geometry=geometry,
        predictions=predictions,
        X_masked=X_masked,
        Y_masked=Y_masked,
        grid_shape=grid_shape,
        mask=mask,
        save_path=os.path.join(output_dir, 'results.png')
    )

    # ========================
    # 8. Save Model
    # ========================
    print("\n" + "="*60)
    print("STEP 8: Saving Model")
    print("="*60)

    model_path = os.path.join(output_dir, 'slope_pinn_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'layers': layers,
        'material_properties': {
            'E': E,
            'nu': nu,
            'rho': rho,
            'g': g
        },
        'geometry': {
            'crest_point': geometry.crest_point.tolist(),
            'toe_point': geometry.toe_point.tolist(),
            'base_height': geometry.base_height,
            'left_boundary': geometry.left_boundary,
            'right_boundary': geometry.right_boundary
        }
    }, model_path)

    print(f"Model saved to: {model_path}")

    # ========================
    # 9. Print Summary Statistics
    # ========================
    print("\n" + "="*60)
    print("STEP 9: Summary Statistics")
    print("="*60)

    print(f"\nDisplacement Statistics:")
    print(f"  Max horizontal displacement (u): {np.max(np.abs(predictions['u'])):.6f} m")
    print(f"  Max vertical displacement (v): {np.max(np.abs(predictions['v'])):.6f} m")
    print(f"  Max total displacement: {np.max(np.sqrt(predictions['u']**2 + predictions['v']**2)):.6f} m")

    print(f"\nStress Statistics:")
    print(f"  Max σ₁₁: {np.max(predictions['sigma_11'])/1e6:.6f} MPa")
    print(f"  Min σ₁₁: {np.min(predictions['sigma_11'])/1e6:.6f} MPa")
    print(f"  Max σ₂₂: {np.max(predictions['sigma_22'])/1e6:.6f} MPa")
    print(f"  Min σ₂₂: {np.min(predictions['sigma_22'])/1e6:.6f} MPa")
    print(f"  Max σ₁₂: {np.max(np.abs(predictions['sigma_12']))/1e6:.6f} MPa")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")


def load_and_predict(model_path: str, geometry: SlopeGeometry):
    """
    Load a saved model and make predictions.

    Args:
        model_path: Path to saved model
        geometry: SlopeGeometry object

    Returns:
        predictions: Dictionary of predicted fields
    """
    # Load checkpoint
    checkpoint = torch.load(model_path)

    # Recreate model
    model = SlopeStabilityPINN(
        layers=checkpoint['layers'],
        E=checkpoint['material_properties']['E'],
        nu=checkpoint['material_properties']['nu'],
        rho=checkpoint['material_properties']['rho'],
        g=checkpoint['material_properties']['g']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = SlopeStabilityTrainer(model=model, device=device)

    # Create prediction grid
    X_masked, Y_masked, grid_shape, mask = create_prediction_grid(geometry)

    # Convert to tensors
    X_tensor = torch.tensor(X_masked.reshape(-1, 1), dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y_masked.reshape(-1, 1), dtype=torch.float32).to(device)

    # Predict
    predictions = trainer.predict(X_tensor, Y_tensor)

    return predictions, X_masked, Y_masked, grid_shape, mask


if __name__ == "__main__":
    main()
