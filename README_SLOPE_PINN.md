# Physics-Informed Neural Network (PINN) for Slope Stability Analysis

A comprehensive implementation of Physics-Informed Neural Networks (PINNs) for analyzing slope stability problems in geotechnical engineering.

## 📋 Overview

This project uses deep learning to solve partial differential equations (PDEs) governing slope stability. The PINN learns displacement and stress fields by enforcing:

- **Equilibrium equations** (force balance)
- **Constitutive relations** (stress-strain relationships)
- **Boundary conditions** (fixed base, free surface, lateral boundaries)

## 🎯 Features

- **Physics-based learning**: Incorporates governing equations directly into the loss function
- **Automatic differentiation**: Computes gradients and derivatives efficiently using PyTorch
- **Flexible geometry**: Easily customize slope dimensions and properties
- **Comprehensive visualization**: Plots displacement, stress fields, and training metrics
- **Material properties**: Configurable Young's modulus, Poisson's ratio, density

## 📁 Project Structure

```
.
├── slope_stability_pinn.py    # Core PINN model and trainer classes
├── slope_geometry.py          # Geometry definition and sampling utilities
├── train_slope_pinn.py        # Main training script
├── requirements_pinn.txt      # Python dependencies
└── README_SLOPE_PINN.md       # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_pinn.txt
   ```

3. **For GPU support** (optional but recommended):
   - Install CUDA toolkit compatible with your PyTorch version
   - Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Running the Code

**Train the model**:
```bash
python train_slope_pinn.py
```

This will:
1. Define the slope geometry
2. Initialize the PINN model
3. Train the network (default: 10,000 epochs)
4. Generate visualizations
5. Save results to a timestamped directory

## 🔧 Customization

### Modify Slope Geometry

Edit the geometry parameters in `train_slope_pinn.py`:

```python
geometry = SlopeGeometry(
    crest_point=(8.0, 10.0),     # Crest coordinates (x, y) in meters
    toe_point=(18.0, 5.0),        # Toe coordinates (x, y) in meters
    base_height=0.0,              # Base elevation
    left_boundary=0.0,            # Left boundary x-coordinate
    right_boundary=20.0           # Right boundary x-coordinate
)
```

### Adjust Material Properties

Modify soil properties in `train_slope_pinn.py`:

```python
E = 10e6        # Young's modulus (Pa)
nu = 0.3        # Poisson's ratio (dimensionless)
rho = 2000.0    # Density (kg/m³)
g = 9.81        # Gravitational acceleration (m/s²)
```

### Network Architecture

Customize the neural network structure:

```python
layers = [2, 64, 64, 64, 64, 2]  # [input_dim, hidden_layers..., output_dim]
```

### Training Hyperparameters

Adjust training settings:

```python
epochs = 10000       # Number of training iterations
lr = 1e-3           # Learning rate
lambda_pde = 1.0    # Weight for PDE loss
lambda_bc = 100.0   # Weight for boundary condition loss
n_domain = 5000     # Number of collocation points
n_boundary = 200    # Boundary points per boundary
```

## 📊 Output

The training script creates a results directory containing:

- `geometry.png` - Visualization of slope geometry with collocation points
- `loss_history.png` - Training loss curves (total, PDE, BC losses)
- `results.png` - Predicted displacement and stress fields
- `slope_pinn_model.pth` - Saved model checkpoint

### Example Outputs

**Predicted Fields:**
- Horizontal displacement (u)
- Vertical displacement (v)
- Displacement magnitude
- Stress components (σ₁₁, σ₂₂, σ₁₂)

## 🧮 Mathematical Formulation

### Governing Equations

**Equilibrium (Force Balance):**
```
∂σ₁₁/∂x + ∂σ₁₂/∂y = 0
∂σ₁₂/∂x + ∂σ₂₂/∂y - ρg = 0
```

**Strain-Displacement Relations:**
```
ε₁₁ = ∂u/∂x
ε₂₂ = ∂v/∂y
ε₁₂ = 0.5(∂u/∂y + ∂v/∂x)
```

**Constitutive Law (Plane Strain Elasticity):**
```
σ₁₁ = (λ + 2μ)ε₁₁ + λε₂₂
σ₂₂ = λε₁₁ + (λ + 2μ)ε₂₂
σ₁₂ = 2με₁₂
```

where:
- λ, μ are Lamé parameters computed from E and ν
- λ = Eν/((1+ν)(1-2ν))
- μ = E/(2(1+ν))

### Boundary Conditions

- **Base (y=0)**: Fixed (u=0, v=0)
- **Left boundary (x=0)**: Roller (u=0)
- **Right boundary (x=20)**: Roller (u=0)
- **Slope surface**: Free (traction-free)

### Loss Function

```
Loss = λ_PDE × Loss_PDE + λ_BC × Loss_BC
```

where:
- **Loss_PDE**: Mean squared residual of equilibrium equations
- **Loss_BC**: Mean squared error of boundary conditions

## 🔬 Advanced Usage

### Loading and Using Trained Models

```python
from train_slope_pinn import load_and_predict
from slope_geometry import SlopeGeometry, plot_results

# Define geometry
geometry = SlopeGeometry(...)

# Load model and predict
predictions, X, Y, shape, mask = load_and_predict(
    model_path='results_slope_pinn_*/slope_pinn_model.pth',
    geometry=geometry
)

# Visualize
plot_results(geometry, predictions, X, Y, shape, mask)
```

### Sampling Methods

The code supports different collocation point sampling strategies:

```python
# Uniform random sampling (default)
x, y = geometry.sample_domain_points(n_points=5000, method='uniform')

# Latin Hypercube Sampling (better space-filling)
x, y = geometry.sample_domain_points(n_points=5000, method='latin_hypercube')
```

## 📚 Theory and Background

### What is a PINN?

Physics-Informed Neural Networks (PINNs) are neural networks trained to solve supervised learning tasks while respecting physical laws described by PDEs. Key advantages:

- **Data efficiency**: Can train with limited or no data
- **Physics consistency**: Solutions automatically satisfy governing equations
- **Mesh-free**: No need for traditional finite element meshing
- **Differentiable**: Enables gradient-based optimization

### Why PINNs for Slope Stability?

Traditional methods (FEM, FDM) require:
- Mesh generation (time-consuming for complex geometries)
- Numerical integration schemes
- Explicit handling of boundary conditions

PINNs offer:
- **Mesh-free formulation**: Points sampled anywhere in domain
- **Automatic differentiation**: Exact derivatives via autograd
- **Flexible boundary conditions**: Incorporated in loss function
- **Continuous solution**: Query predictions at any point

## ⚙️ Performance Tips

1. **GPU Acceleration**: Train on GPU for 10-100× speedup
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. **Batch Processing**: Use mini-batches for large datasets

3. **Learning Rate Scheduling**: Implemented via `ReduceLROnPlateau`

4. **Network Depth**: Deeper networks (4-6 hidden layers) often work better

5. **Collocation Points**: More points = better accuracy but slower training

## 🐛 Troubleshooting

**High Loss Values:**
- Increase `lambda_bc` to enforce boundary conditions more strongly
- Check geometry and boundary condition definitions
- Increase number of training epochs

**NaN/Inf Losses:**
- Reduce learning rate
- Check material property values (avoid extreme values)
- Enable gradient clipping

**Slow Training:**
- Reduce number of collocation points
- Use GPU acceleration
- Simplify network architecture

## 📖 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Haghighat, E., Raissi, M., Moure, A., Gomez, H., & Juanes, R. (2021). A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. *Computer Methods in Applied Mechanics and Engineering*, 379, 113741.

3. Zhang, E., Dao, M., Karniadakis, G. E., & Suresh, S. (2022). Analyses of internal structures and defects in materials using physics-informed neural networks. *Science Advances*, 8(7), eabk0644.

## 📝 License

This project is provided for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add plasticity models (Mohr-Coulomb, Drucker-Prager)
- [ ] Implement time-dependent analysis
- [ ] Add water table and pore pressure effects
- [ ] Support for layered soils
- [ ] Uncertainty quantification
- [ ] Transfer learning for different geometries

## 💬 Contact

For questions or feedback, please open an issue or contact the repository maintainer.

---

**Note**: This implementation is for educational purposes. For critical engineering applications, validate results against established numerical methods and consider consulting with geotechnical engineers.
