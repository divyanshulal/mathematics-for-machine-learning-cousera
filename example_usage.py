"""
Example Usage Scripts for Slope Stability PINN

This module demonstrates various ways to use the PINN for slope stability analysis.
"""

import torch
import numpy as np
from slope_stability_pinn import SlopeStabilityPINN, SlopeStabilityTrainer
from slope_geometry import SlopeGeometry, prepare_training_data, create_prediction_grid, plot_results


# ============================================================================
# Example 1: Quick Training with Default Parameters
# ============================================================================

def example_1_quick_training():
    """
    Quick training example with default parameters.
    Useful for rapid prototyping and testing.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Training (1000 epochs)")
    print("="*70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    geometry = SlopeGeometry(
        crest_point=(8.0, 10.0),
        toe_point=(18.0, 5.0),
        base_height=0.0,
        left_boundary=0.0,
        right_boundary=20.0
    )

    # Initialize model
    model = SlopeStabilityPINN(
        layers=[2, 32, 32, 32, 2],  # Smaller network for faster training
        E=10e6,
        nu=0.3,
        rho=2000.0,
        g=9.81
    )

    # Prepare data
    training_data = prepare_training_data(geometry, n_domain=2000, n_boundary=100, device=device)

    # Train
    trainer = SlopeStabilityTrainer(model, device)
    trainer.train(
        x_domain=training_data['x_domain'],
        y_domain=training_data['y_domain'],
        x_bc=training_data['x_bc'],
        y_bc=training_data['y_bc'],
        bc_values=training_data['bc_values'],
        epochs=1000,
        lr=1e-3,
        lambda_pde=1.0,
        lambda_bc=100.0,
        print_every=200
    )

    print("Training completed!")
    return trainer, geometry


# ============================================================================
# Example 2: Custom Slope Geometry (Steep Slope)
# ============================================================================

def example_2_steep_slope():
    """
    Analyze a steeper slope with different geometry.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Steep Slope Analysis")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Steeper slope: 45-degree angle
    geometry = SlopeGeometry(
        crest_point=(5.0, 10.0),     # Moved crest left
        toe_point=(15.0, 3.0),        # Lower toe height
        base_height=0.0,
        left_boundary=0.0,
        right_boundary=20.0
    )

    # Visualize geometry
    geometry.plot_geometry(show_points=True, n_points=1000)

    # Rest of training code...
    print("Steep slope geometry defined. Angle: ~40 degrees")
    return geometry


# ============================================================================
# Example 3: Different Material Properties (Soft Clay vs Stiff Clay)
# ============================================================================

def example_3_material_comparison():
    """
    Compare results for different material properties.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Material Properties Comparison")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    geometry = SlopeGeometry()

    materials = {
        'soft_clay': {'E': 5e6, 'nu': 0.35, 'rho': 1800.0, 'name': 'Soft Clay'},
        'stiff_clay': {'E': 20e6, 'nu': 0.25, 'rho': 2100.0, 'name': 'Stiff Clay'},
        'dense_sand': {'E': 50e6, 'nu': 0.3, 'rho': 2000.0, 'name': 'Dense Sand'}
    }

    results = {}

    for mat_key, mat_props in materials.items():
        print(f"\nTraining model for {mat_props['name']}...")
        print(f"  E = {mat_props['E']/1e6:.1f} MPa")
        print(f"  ν = {mat_props['nu']}")
        print(f"  ρ = {mat_props['rho']} kg/m³")

        # Initialize model with specific material properties
        model = SlopeStabilityPINN(
            layers=[2, 64, 64, 64, 2],
            E=mat_props['E'],
            nu=mat_props['nu'],
            rho=mat_props['rho'],
            g=9.81
        )

        # Train (reduced epochs for demonstration)
        trainer = SlopeStabilityTrainer(model, device)
        training_data = prepare_training_data(geometry, n_domain=3000, n_boundary=150, device=device)

        trainer.train(
            x_domain=training_data['x_domain'],
            y_domain=training_data['y_domain'],
            x_bc=training_data['x_bc'],
            y_bc=training_data['y_bc'],
            bc_values=training_data['bc_values'],
            epochs=2000,
            lr=1e-3,
            print_every=500
        )

        results[mat_key] = trainer

    print("\nComparison complete!")
    return results, geometry


# ============================================================================
# Example 4: Progressive Training (Transfer Learning)
# ============================================================================

def example_4_progressive_training():
    """
    Demonstrate progressive training strategy:
    1. Train with coarse resolution
    2. Fine-tune with higher resolution
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Progressive Training Strategy")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    geometry = SlopeGeometry()

    # Initialize model
    model = SlopeStabilityPINN(layers=[2, 64, 64, 64, 2])
    trainer = SlopeStabilityTrainer(model, device)

    # Stage 1: Coarse resolution
    print("\nStage 1: Coarse resolution training...")
    training_data_coarse = prepare_training_data(geometry, n_domain=2000, n_boundary=100, device=device)

    trainer.train(
        x_domain=training_data_coarse['x_domain'],
        y_domain=training_data_coarse['y_domain'],
        x_bc=training_data_coarse['x_bc'],
        y_bc=training_data_coarse['y_bc'],
        bc_values=training_data_coarse['bc_values'],
        epochs=3000,
        lr=1e-3,
        print_every=1000
    )

    # Stage 2: Fine resolution
    print("\nStage 2: Fine resolution training...")
    training_data_fine = prepare_training_data(geometry, n_domain=5000, n_boundary=200, device=device)

    trainer.train(
        x_domain=training_data_fine['x_domain'],
        y_domain=training_data_fine['y_domain'],
        x_bc=training_data_fine['x_bc'],
        y_bc=training_data_fine['y_bc'],
        bc_values=training_data_fine['bc_values'],
        epochs=2000,
        lr=5e-4,  # Lower learning rate for fine-tuning
        print_every=500
    )

    print("Progressive training completed!")
    return trainer, geometry


# ============================================================================
# Example 5: Query Specific Points
# ============================================================================

def example_5_query_points(trainer, geometry):
    """
    Query predictions at specific points of interest.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Query Specific Points")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define points of interest
    points_of_interest = {
        'Crest': (8.0, 10.0),
        'Mid-slope': (13.0, 7.5),
        'Toe': (18.0, 5.0),
        'Base center': (10.0, 0.5),
        'Deep interior': (10.0, 3.0)
    }

    print("\nPredictions at key locations:")
    print("-" * 70)

    for name, (x_val, y_val) in points_of_interest.items():
        # Convert to tensor
        x = torch.tensor([[x_val]], dtype=torch.float32).to(device)
        y = torch.tensor([[y_val]], dtype=torch.float32).to(device)

        # Predict
        predictions = trainer.predict(x, y)

        print(f"\n{name} (x={x_val:.1f}m, y={y_val:.1f}m):")
        print(f"  Displacement u: {predictions['u'][0, 0]:.6e} m")
        print(f"  Displacement v: {predictions['v'][0, 0]:.6e} m")
        print(f"  Stress σ₁₁: {predictions['sigma_11'][0, 0]/1e6:.6f} MPa")
        print(f"  Stress σ₂₂: {predictions['sigma_22'][0, 0]/1e6:.6f} MPa")
        print(f"  Stress σ₁₂: {predictions['sigma_12'][0, 0]/1e6:.6f} MPa")


# ============================================================================
# Example 6: Batch Prediction on Grid
# ============================================================================

def example_6_batch_prediction(trainer, geometry):
    """
    Make predictions on a regular grid for visualization.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Batch Prediction on Grid")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create prediction grid
    X_masked, Y_masked, grid_shape, mask = create_prediction_grid(
        geometry=geometry,
        nx=150,  # Higher resolution
        ny=75
    )

    print(f"Grid resolution: {grid_shape}")
    print(f"Total grid points: {grid_shape[0] * grid_shape[1]}")
    print(f"Valid points in domain: {len(X_masked)}")

    # Convert to tensors
    X_tensor = torch.tensor(X_masked.reshape(-1, 1), dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y_masked.reshape(-1, 1), dtype=torch.float32).to(device)

    # Predict
    print("Computing predictions...")
    predictions = trainer.predict(X_tensor, Y_tensor)

    # Plot results
    print("Plotting results...")
    plot_results(geometry, predictions, X_masked, Y_masked, grid_shape, mask)

    return predictions


# ============================================================================
# Example 7: Sensitivity Analysis
# ============================================================================

def example_7_sensitivity_analysis():
    """
    Perform sensitivity analysis on slope angle.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Slope Angle Sensitivity Analysis")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Different slope angles
    slope_configs = [
        {'toe_x': 20.0, 'toe_y': 7.0, 'angle': '~20°'},
        {'toe_x': 18.0, 'toe_y': 5.0, 'angle': '~30°'},
        {'toe_x': 15.0, 'toe_y': 3.0, 'angle': '~40°'},
    ]

    results = {}

    for config in slope_configs:
        print(f"\nAnalyzing slope with angle {config['angle']}...")

        geometry = SlopeGeometry(
            crest_point=(8.0, 10.0),
            toe_point=(config['toe_x'], config['toe_y']),
            base_height=0.0,
            left_boundary=0.0,
            right_boundary=20.0
        )

        # Quick training
        model = SlopeStabilityPINN(layers=[2, 48, 48, 48, 2])
        trainer = SlopeStabilityTrainer(model, device)
        training_data = prepare_training_data(geometry, n_domain=3000, n_boundary=150, device=device)

        trainer.train(
            x_domain=training_data['x_domain'],
            y_domain=training_data['y_domain'],
            x_bc=training_data['x_bc'],
            y_bc=training_data['y_bc'],
            bc_values=training_data['bc_values'],
            epochs=2000,
            lr=1e-3,
            print_every=500
        )

        results[config['angle']] = {'trainer': trainer, 'geometry': geometry}

    print("\nSensitivity analysis complete!")
    return results


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run selected examples."""
    print("="*70)
    print("SLOPE STABILITY PINN - EXAMPLE USAGE")
    print("="*70)

    # Uncomment the examples you want to run:

    # Example 1: Quick training
    # trainer, geometry = example_1_quick_training()

    # Example 2: Steep slope
    # geometry = example_2_steep_slope()

    # Example 3: Material comparison
    # results, geometry = example_3_material_comparison()

    # Example 4: Progressive training
    # trainer, geometry = example_4_progressive_training()

    # Example 5: Query specific points (requires trained model)
    # example_5_query_points(trainer, geometry)

    # Example 6: Batch prediction (requires trained model)
    # example_6_batch_prediction(trainer, geometry)

    # Example 7: Sensitivity analysis
    # results = example_7_sensitivity_analysis()

    print("\n" + "="*70)
    print("To run examples, uncomment them in the main() function.")
    print("="*70)


if __name__ == "__main__":
    main()
