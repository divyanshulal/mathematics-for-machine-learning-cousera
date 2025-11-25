"""
Minor Creek PINN - CORRECTED VERSION
=====================================
Key fixes:
1. Displacement computed FROM physics (not directly predicted by network)
2. Physics parameters directly affect loss through viscoplastic law
3. Proper unit handling with α having units [1/kPa]
4. Network only predicts pore pressure, displacement follows from physics
5. FIXED: Gradient flow through integration using torch.cat instead of in-place assignment
6. FIXED: Proper tensor shape handling throughout

Based on: Li, Handwerger, Buscarnera (2023) - Landslides 20:1101-1113
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# PHYSICAL CONSTANTS (FROM PAPER TABLE 2)
# ============================================================================
class PhysicalConstants:
    """Physical constants from Li et al. (2023) for Minor Creek"""

    # Geometry
    H = 6.0  # Landslide thickness [m]
    H_shear = 1.0  # Shear zone thickness [m]
    theta_deg = 15.0  # Slope angle [degrees]
    theta_rad = np.radians(15.0)

    # Material properties
    gamma_sat = 22.0  # Saturated unit weight [kN/m³]
    gamma_w = 9.81  # Unit weight of water [kN/m³]

    # Stress state at shear zone (z = H = 6m)
    # σ_total = γ_sat * H * cos²θ = 22 * 6 * cos²(15°) ≈ 123.1 kPa
    # τ = γ_sat * H * sinθ * cosθ = 22 * 6 * sin(15°) * cos(15°) ≈ 33.0 kPa
    sigma_total = gamma_sat * H * np.cos(theta_rad) ** 2  # ≈ 123.1 kPa
    tau = gamma_sat * H * np.sin(theta_rad) * np.cos(theta_rad)  # ≈ 33.0 kPa

    # Target parameters from Table 2
    ks_target = 4.45e-6  # [m/s]
    Ss_target = 0.27  # [1/m]
    phi_target = 18.9  # [degrees]
    alpha_target = 8.0  # [1/kPa] - NOTE: paper uses dimensionless, but physically should be 1/kPa
    mu_target = 2.1e-8  # [(kPa·s)^-1]

    # Reference scales
    p_ref = 10.0  # Reference pore pressure [kPa]
    T_ref = 365.25  # Reference time [days]

    @classmethod
    def print_stress_state(cls):
        """Print stress state for verification"""
        print(f"\n📐 STRESS STATE AT SHEAR ZONE (z = {cls.H}m):")
        print(f"  σ_total = {cls.sigma_total:.1f} kPa")
        print(f"  τ = {cls.tau:.1f} kPa")
        print(f"  τ/σ_total = {cls.tau / cls.sigma_total:.3f}")
        print(f"  Equivalent angle = {np.degrees(np.arctan(cls.tau / cls.sigma_total)):.1f}°")


# ============================================================================
# DATA LOADING
# ============================================================================
def load_minor_creek_data():
    """Load Minor Creek data (1982-1985) digitized from paper"""

    # Pore pressure data (digitized from Fig. 6a)
    data_csv = """1982.00,0.0
1982.03,1.2
1982.07,3.5
1982.10,5.8
1982.14,7.5
1982.17,8.0
1982.21,9.5
1982.24,10.2
1982.28,10.5
1982.31,10.8
1982.34,10.5
1982.38,9.8
1982.41,9.0
1982.45,8.5
1982.48,8.0
1982.52,7.5
1982.55,7.0
1982.59,6.5
1982.62,6.2
1982.66,5.8
1982.69,5.5
1982.72,5.2
1982.76,5.0
1982.79,4.8
1982.83,4.6
1982.86,4.5
1982.90,4.5
1982.93,4.2
1982.97,3.8
1983.00,3.5
1983.03,4.0
1983.07,6.5
1983.10,8.0
1983.14,9.0
1983.17,9.8
1983.21,10.0
1983.24,9.5
1983.28,9.2
1983.31,9.0
1983.34,9.5
1983.38,10.0
1983.41,9.8
1983.45,9.5
1983.48,9.0
1983.52,8.5
1983.55,8.0
1983.59,7.5
1983.62,7.0
1983.66,6.5
1983.69,6.0
1983.72,5.5
1983.76,5.0
1983.79,4.5
1983.83,4.0
1983.86,3.5
1983.90,3.2
1983.93,3.0
1983.97,3.5
1984.00,5.0
1984.03,8.0
1984.07,9.5
1984.10,10.2
1984.14,10.0
1984.17,9.5
1984.21,9.2
1984.24,9.0
1984.28,9.2
1984.31,9.5
1984.34,9.8
1984.38,9.5
1984.41,9.0
1984.45,8.5
1984.48,8.0
1984.52,8.2
1984.55,8.5
1984.59,8.0
1984.62,7.5
1984.66,7.2
1984.69,8.0
1984.72,9.0
1984.76,9.2
1984.79,8.5
1984.83,8.0
1984.86,8.2
1984.90,8.5
1984.93,8.0
1984.97,7.5
1985.00,7.2
1985.03,7.0
1985.07,6.8
1985.10,6.5
1985.14,6.2
1985.17,6.0
1985.21,5.8
1985.24,5.5
1985.28,5.2
1985.31,5.0
1985.34,4.5
1985.38,4.0
1985.41,3.0"""

    lines = data_csv.strip().split('\n')
    years, pore_pressure = [], []
    for line in lines:
        parts = line.split(',')
        years.append(float(parts[0]))
        pore_pressure.append(float(parts[1]))

    years = np.array(years)
    pore_pressure = np.array(pore_pressure)  # [kPa] - this is Δp_w from paper
    t_days = (years - 1982.0) * 365.25  # Convert to days from start

    # Displacement data from Fig. 6b (digitized)
    # Note: These are cumulative displacements
    disp_times = np.array([0, 60, 120, 180, 240, 300, 365, 425, 485, 545, 600,
                           650, 730, 850, 950, 1095, 1200])  # days
    disp_vals = np.array([0.0, 0.05, 0.15, 0.20, 0.20, 0.20, 0.22, 0.40, 0.80,
                          0.85, 0.85, 0.87, 0.90, 1.10, 1.20, 1.25, 1.28])  # m

    print(f"\n📊 DATA LOADED:")
    print(f"  Pore pressure points: {len(pore_pressure)}")
    print(f"  Displacement points: {len(disp_times)}")
    print(f"  Time range: {t_days[0]:.0f} - {t_days[-1]:.0f} days ({(t_days[-1] - t_days[0]) / 365.25:.1f} years)")
    print(f"  Pore pressure range: {pore_pressure.min():.1f} - {pore_pressure.max():.1f} kPa")
    print(f"  Displacement range: {disp_vals.min():.2f} - {disp_vals.max():.2f} m")

    P = PhysicalConstants

    return {
        # Dimensional
        't_days': t_days,
        'pore_pressure': pore_pressure,
        't_disp_days': disp_times,
        'displacement': disp_vals,

        # Non-dimensional (for neural network)
        't_star': torch.tensor(t_days / P.T_ref, dtype=torch.float32, device=device).reshape(-1, 1),
        'pw_star': torch.tensor(pore_pressure / P.p_ref, dtype=torch.float32, device=device).reshape(-1, 1),
        't_disp_star': torch.tensor(disp_times / P.T_ref, dtype=torch.float32, device=device).reshape(-1, 1),

        # Interpolation for pore pressure (for computing displacement)
        'pw_interp': interp1d(t_days, pore_pressure, kind='linear',
                              bounds_error=False, fill_value=(pore_pressure[0], pore_pressure[-1])),
    }


# ============================================================================
# NEURAL NETWORK - PREDICTS ONLY PORE PRESSURE
# ============================================================================
class PorePressureNet(nn.Module):
    """
    Neural network that predicts pore pressure only.
    Displacement is computed through physics integration.
    """

    def __init__(self, hidden_units=64, hidden_layers=4):
        super().__init__()

        # Build network
        layers = [nn.Linear(1, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
        layers.append(nn.Linear(hidden_units, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, t_star):
        """
        Predict normalized pore pressure change
        Input: t_star = t / T_ref
        Output: pw_star = Δp_w / p_ref
        """
        # Normalize input to [-1, 1] range
        t_norm = 2.0 * t_star / 3.5 - 1.0  # ~3.5 years of data
        return self.net(t_norm)


# ============================================================================
# PHYSICS-INFORMED CALIBRATION MODEL
# ============================================================================
class LandslidePINN(nn.Module):
    """
    Complete PINN model for landslide calibration.

    Key insight: Displacement is NOT predicted directly by network.
    Instead, it's computed from physics using calibrated parameters.

    Physics:
    - Pore pressure predicted by neural network (fits data)
    - Strain rate computed from viscoplastic law (Eq. 6)
    - Displacement from integrating strain rate
    """

    def __init__(self, hidden_units=64, hidden_layers=4):
        super().__init__()

        self.P = PhysicalConstants

        # Neural network for pore pressure
        self.pw_net = PorePressureNet(hidden_units, hidden_layers)

        # Learnable physics parameters (in transformed space for stability)
        # phi in degrees, stored as log for positivity
        self.log_phi = nn.Parameter(torch.tensor(np.log(20.0), device=device))

        # mu in [(kPa·s)^-1], stored as log
        self.log_mu = nn.Parameter(torch.tensor(np.log(1e-7), device=device))

        # alpha in [1/kPa], stored as log
        # Note: α*f should be dimensionless, so α has units 1/kPa
        self.log_alpha = nn.Parameter(torch.tensor(np.log(1.0), device=device))

        # Hydraulic parameters (for future extension)
        self.log_ks = nn.Parameter(torch.tensor(np.log(1e-5), device=device))
        self.log_Ss = nn.Parameter(torch.tensor(np.log(0.3), device=device))

        print(f"\n🧠 PINN ARCHITECTURE:")
        print(f"  Pore pressure network: {hidden_layers} layers × {hidden_units} units")
        print(f"  Physics parameters: φ, μ, α, ks, Ss (5 learnable)")

    def get_params(self):
        """Get physics parameters with soft bounds via sigmoid transformation"""
        # Friction angle: 10° to 35°
        phi = 10.0 + 25.0 * torch.sigmoid(self.log_phi - np.log(20.0))

        # Viscosity: 1e-9 to 1e-6 (kPa·s)^-1
        mu = torch.exp(self.log_mu)
        mu = torch.clamp(mu, 1e-10, 1e-5)

        # Alpha: 0.1 to 20 [1/kPa]
        alpha = torch.exp(self.log_alpha)
        alpha = torch.clamp(alpha, 0.01, 50.0)

        # Hydraulic
        ks = torch.exp(self.log_ks)
        Ss = torch.exp(self.log_Ss)

        return {'phi': phi, 'mu': mu, 'alpha': alpha, 'ks': ks, 'Ss': Ss}

    def get_params_numpy(self):
        params = self.get_params()
        return {k: v.detach().cpu().item() for k, v in params.items()}

    def predict_pore_pressure(self, t_star):
        """Predict pore pressure using neural network"""
        return self.pw_net(t_star) * self.P.p_ref  # Return in kPa

    def compute_strain_rate(self, pw):
        """
        Compute viscoplastic strain rate using hybrid law (Eq. 6)

        Physics:
        - Effective stress: σ' = σ_total - p_w
        - Yield function: f = τ - tan(φ) × σ' [kPa]
        - Hybrid law: γ̇ = μ × ln(1 + exp(α × f))

        Args:
            pw: pore water pressure [kPa]

        Returns:
            gamma_dot: strain rate [1/s]
            f: yield function value [kPa]
        """
        params = self.get_params()
        phi_rad = params['phi'] * np.pi / 180.0
        mu = params['mu']  # [(kPa·s)^-1]
        alpha = params['alpha']  # [1/kPa]

        # Effective stress at shear zone
        sigma_eff = self.P.sigma_total - pw  # [kPa]
        sigma_eff = torch.clamp(sigma_eff, min=1.0)  # Prevent negative

        # Yield function [kPa]
        f = self.P.tau - torch.tan(phi_rad) * sigma_eff

        # Hybrid viscoplastic law (Eq. 6)
        # γ̇ = μ × ln(1 + exp(α × f))
        # Note: α × f should be dimensionless → α has units [1/kPa]
        alpha_f = alpha * f  # dimensionless
        alpha_f = torch.clamp(alpha_f, -20.0, 20.0)  # Numerical stability

        gamma_dot = mu * torch.log(1.0 + torch.exp(alpha_f))  # [1/s]

        return gamma_dot, f

    def compute_displacement(self, t_tensor, pw_tensor):
        """
        Compute displacement by integrating strain rate.

        u(t) = H_shear × ∫₀ᵗ γ̇(t') dt'

        Uses trapezoidal integration.

        FIX: Uses torch.cat instead of in-place assignment to maintain gradient flow.
        """
        # Compute strain rate at each time point
        gamma_dot, f = self.compute_strain_rate(pw_tensor)  # [1/s]

        # Convert to per day
        gamma_dot_day = gamma_dot * 86400.0  # [1/day]

        # Displacement rate [m/day]
        du_dt = gamma_dot_day * self.P.H_shear

        # Get time in days
        t_days = t_tensor * self.P.T_ref

        # Flatten tensors for integration
        t_flat = t_days.flatten()
        du_dt_flat = du_dt.flatten()

        # Trapezoidal integration
        dt = t_flat[1:] - t_flat[:-1]
        avg_rate = 0.5 * (du_dt_flat[1:] + du_dt_flat[:-1])

        # Cumulative sum (cumtrapz equivalent)
        increments = avg_rate * dt

        # FIX: Use torch.cat instead of in-place assignment to maintain gradient flow
        # This ensures gradients properly flow back to physics parameters
        zero = torch.zeros(1, device=t_tensor.device, dtype=t_tensor.dtype)
        u = torch.cat([zero, torch.cumsum(increments, dim=0)])

        return u, gamma_dot.flatten(), f.flatten()

    def forward(self, t_star, return_details=False):
        """
        Forward pass: predict pore pressure and compute displacement

        This is the key difference from original: displacement comes from physics!
        """
        pw = self.predict_pore_pressure(t_star)
        u, gamma_dot, f = self.compute_displacement(t_star, pw)

        if return_details:
            return pw, u, gamma_dot, f
        return pw, u


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_pinn(epochs=10000, lr_net=1e-3, lr_params=1e-2, print_every=500):
    """
    Train PINN with physics-based displacement computation.

    Key insight: The network learns pore pressure from data.
    Physics parameters are calibrated to match displacement.
    """
    print("\n" + "=" * 80)
    print("  TRAINING PHYSICS-INFORMED NEURAL NETWORK")
    print("  Displacement computed FROM physics, not predicted directly")
    print("=" * 80)

    # Load data
    data = load_minor_creek_data()
    PhysicalConstants.print_stress_state()

    # Initialize model
    model = LandslidePINN(hidden_units=64, hidden_layers=4).to(device)

    # Record initial parameters
    params_init = model.get_params_numpy()
    print(f"\n📍 INITIAL PARAMETERS:")
    for k, v in params_init.items():
        if k in ['mu', 'ks']:
            print(f"  {k}: {v:.2e}")
        else:
            print(f"  {k}: {v:.3f}")

    # Separate optimizers with different learning rates
    net_params = list(model.pw_net.parameters())
    phys_params = [model.log_phi, model.log_mu, model.log_alpha,
                   model.log_ks, model.log_Ss]

    opt_net = torch.optim.Adam(net_params, lr=lr_net)
    opt_phys = torch.optim.Adam(phys_params, lr=lr_params)

    # History
    history = {
        'loss_total': [], 'loss_pw': [], 'loss_u': [],
        'phi': [], 'mu': [], 'alpha': [], 'ks': [], 'Ss': [],
        'f_mean': [], 'gamma_dot_mean': [],
        'grad_phi': [], 'grad_mu': [], 'grad_alpha': [],
    }

    print(f"\n{'Epoch':>6} | {'Total':>10} | {'PW Loss':>10} | {'U Loss':>10} | "
          f"{'φ':>6} | {'μ':>10} | {'α':>6} | {'∂L/∂φ':>10}")
    print("-" * 95)

    # Displacement observations (convert to tensor)
    t_disp_obs = data['t_disp_star'].to(device)
    u_obs = torch.tensor(data['displacement'], dtype=torch.float32, device=device)

    for epoch in range(epochs):
        model.train()
        opt_net.zero_grad()
        opt_phys.zero_grad()

        # ====================================================================
        # LOSS 1: Pore pressure data fitting
        # ====================================================================
        pw_pred = model.predict_pore_pressure(data['t_star'])
        pw_obs = data['pw_star'] * PhysicalConstants.p_ref  # Convert to kPa
        loss_pw = torch.mean((pw_pred - pw_obs) ** 2)

        # ====================================================================
        # LOSS 2: Displacement through physics (THE KEY LOSS!)
        # ====================================================================
        # Predict at displacement observation times
        pw_at_disp = model.predict_pore_pressure(t_disp_obs)
        u_pred, gamma_dot, f = model.compute_displacement(t_disp_obs, pw_at_disp)

        # FIX: Ensure shapes match for loss computation
        # u_pred is already flattened from compute_displacement
        # u_obs needs to be the same shape
        loss_u = torch.mean((u_pred - u_obs) ** 2)

        # ====================================================================
        # TOTAL LOSS with curriculum
        # ====================================================================
        if epoch < 1000:
            # Phase 1: Focus on pore pressure fitting
            w_pw = 1.0
            w_u = 0.1
        elif epoch < 5000:
            # Phase 2: Gradually increase displacement weight
            progress = (epoch - 1000) / 4000
            w_pw = 1.0
            w_u = 0.1 + progress * 9.9  # 0.1 → 10
        else:
            # Phase 3: Full physics
            w_pw = 1.0
            w_u = 10.0

        loss = w_pw * loss_pw + w_u * loss_u

        # Backward
        loss.backward()

        # Check gradients before clipping
        grad_phi = model.log_phi.grad.item() if model.log_phi.grad is not None else 0.0
        grad_mu = model.log_mu.grad.item() if model.log_mu.grad is not None else 0.0
        grad_alpha = model.log_alpha.grad.item() if model.log_alpha.grad is not None else 0.0

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(net_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(phys_params, max_norm=1.0)

        # Update
        opt_net.step()
        opt_phys.step()  # Always update physics params

        # Record history
        params = model.get_params_numpy()
        history['loss_total'].append(loss.item())
        history['loss_pw'].append(loss_pw.item())
        history['loss_u'].append(loss_u.item())
        for k in ['phi', 'mu', 'alpha', 'ks', 'Ss']:
            history[k].append(params[k])
        history['f_mean'].append(f.mean().item())
        history['gamma_dot_mean'].append(gamma_dot.mean().item())
        history['grad_phi'].append(grad_phi)
        history['grad_mu'].append(grad_mu)
        history['grad_alpha'].append(grad_alpha)

        # Print progress
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"{epoch:>6} | {loss.item():>10.2e} | {loss_pw.item():>10.2e} | "
                  f"{loss_u.item():>10.2e} | {params['phi']:>6.2f} | "
                  f"{params['mu']:>10.2e} | {params['alpha']:>6.2f} | {grad_phi:>10.2e}")

    params_final = model.get_params_numpy()

    # Print results
    print("\n" + "=" * 80)
    print("  CALIBRATION RESULTS")
    print("=" * 80)

    P = PhysicalConstants
    targets = {
        'phi': P.phi_target, 'mu': P.mu_target, 'alpha': P.alpha_target,
        'ks': P.ks_target, 'Ss': P.Ss_target,
    }

    print(f"\n📍 FINAL PARAMETERS vs TARGET (Table 2):")
    print(f"  {'Param':<8} {'Initial':>12} {'Final':>12} {'Target':>12} {'Error':>10}")
    print("-" * 60)

    for k in ['phi', 'mu', 'alpha', 'ks', 'Ss']:
        init = params_init[k]
        final = params_final[k]
        target = targets[k]
        error = abs(final - target) / target * 100

        if k in ['mu', 'ks']:
            print(f"  {k:<8} {init:>12.2e} {final:>12.2e} {target:>12.2e} {error:>9.1f}%")
        else:
            print(f"  {k:<8} {init:>12.3f} {final:>12.3f} {target:>12.3f} {error:>9.1f}%")

    return model, history, data, params_init, params_final


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_figure(model, history, data, params_init, params_final):
    """Create comprehensive visualization"""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    P = PhysicalConstants

    # ========================================================================
    # ROW 1: DATA FITS
    # ========================================================================

    # 1a. Pore Pressure
    ax1 = fig.add_subplot(gs[0, 0])
    with torch.no_grad():
        pw_pred = model.predict_pore_pressure(data['t_star']).cpu().numpy().flatten()
    ax1.plot(data['t_days'], pw_pred, 'b-', linewidth=2, label='PINN')
    ax1.plot(data['t_days'], data['pore_pressure'], 'ro', markersize=3, alpha=0.5, label='Observed')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('ΔPore Pressure (kPa)')
    ax1.set_title('(a) Pore Pressure at Shear Zone')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1b. Displacement
    ax2 = fig.add_subplot(gs[0, 1])
    with torch.no_grad():
        pw_full = model.predict_pore_pressure(data['t_star'])
        u_pred_full, _, _ = model.compute_displacement(data['t_star'], pw_full)
        u_pred_full = u_pred_full.cpu().numpy()

        pw_at_obs = model.predict_pore_pressure(data['t_disp_star'])
        u_pred_obs, _, _ = model.compute_displacement(data['t_disp_star'], pw_at_obs)
        u_pred_obs = u_pred_obs.cpu().numpy()

    ax2.plot(data['t_days'], u_pred_full, 'g-', linewidth=2, label='PINN')
    ax2.plot(data['t_disp_days'], data['displacement'], 'ro', markersize=6, label='Observed')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Displacement (m)')
    ax2.set_title('(b) Surface Displacement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 1c. Strain rate
    ax3 = fig.add_subplot(gs[0, 2])
    with torch.no_grad():
        pw_full = model.predict_pore_pressure(data['t_star'])
        gamma_dot, f = model.compute_strain_rate(pw_full)
        gamma_dot_day = (gamma_dot * 86400).cpu().numpy().flatten()
    ax3.semilogy(data['t_days'], gamma_dot_day, 'purple', linewidth=1.5)
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('γ̇ (1/day)')
    ax3.set_title('(c) Strain Rate')
    ax3.grid(True, alpha=0.3)

    # 1d. Yield function
    ax4 = fig.add_subplot(gs[0, 3])
    f_vals = f.cpu().numpy().flatten()
    ax4.plot(data['t_days'], f_vals, 'brown', linewidth=1.5)
    ax4.axhline(0, color='k', linestyle='--', linewidth=1, label='Yield (f=0)')
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('f (kPa)')
    ax4.set_title('(d) Yield Function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 2: PARAMETER EVOLUTION
    # ========================================================================
    epochs_arr = np.arange(len(history['phi']))

    # 2a. Friction angle
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(epochs_arr, history['phi'], 'b-', linewidth=2)
    ax5.axhline(P.phi_target, color='r', linestyle='--', linewidth=2, label=f'Target={P.phi_target}°')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('φ (degrees)')
    ax5.set_title('(e) Friction Angle')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 2b. Viscosity
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.semilogy(epochs_arr, history['mu'], 'purple', linewidth=2)
    ax6.axhline(P.mu_target, color='r', linestyle='--', linewidth=2, label=f'Target={P.mu_target:.1e}')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('μ [(kPa·s)⁻¹]')
    ax6.set_title('(f) Viscosity Parameter')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 2c. Alpha
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(epochs_arr, history['alpha'], 'brown', linewidth=2)
    ax7.axhline(P.alpha_target, color='r', linestyle='--', linewidth=2, label=f'Target={P.alpha_target}')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('α (1/kPa)')
    ax7.set_title('(g) Flow Parameter')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 2d. Gradients
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(epochs_arr, np.abs(history['grad_phi']), 'b-', alpha=0.7, label='|∂L/∂log_φ|')
    ax8.plot(epochs_arr, np.abs(history['grad_mu']), 'purple', alpha=0.7, label='|∂L/∂log_μ|')
    ax8.plot(epochs_arr, np.abs(history['grad_alpha']), 'brown', alpha=0.7, label='|∂L/∂log_α|')
    ax8.set_yscale('log')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Gradient magnitude')
    ax8.set_title('(h) Parameter Gradients')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # ========================================================================
    # ROW 3: LOSSES AND DIAGNOSTICS
    # ========================================================================

    # 3a. Loss evolution
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.semilogy(epochs_arr, history['loss_total'], 'k-', linewidth=2, label='Total')
    ax9.semilogy(epochs_arr, history['loss_pw'], 'b-', linewidth=1.5, label='Pore Pressure', alpha=0.7)
    ax9.semilogy(epochs_arr, history['loss_u'], 'g-', linewidth=1.5, label='Displacement', alpha=0.7)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Loss')
    ax9.set_title('(i) Loss Evolution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 3b. Mean strain rate evolution
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.semilogy(epochs_arr, history['gamma_dot_mean'], 'purple', linewidth=1.5)
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Mean γ̇ (1/s)')
    ax10.set_title('(j) Strain Rate Evolution')
    ax10.grid(True, alpha=0.3)

    # 3c. Yield function evolution
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.plot(epochs_arr, history['f_mean'], 'brown', linewidth=1.5)
    ax11.axhline(0, color='k', linestyle='--', linewidth=1)
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('Mean f (kPa)')
    ax11.set_title('(k) Yield Function Evolution')
    ax11.grid(True, alpha=0.3)

    # 3d. Summary
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    targets = {
        'phi': P.phi_target, 'mu': P.mu_target, 'alpha': P.alpha_target,
        'ks': P.ks_target, 'Ss': P.Ss_target
    }

    summary = "CALIBRATION RESULTS\n" + "=" * 30 + "\n\n"
    summary += f"{'Param':<6} {'Final':>10} {'Target':>10} {'Err%':>8}\n"
    summary += "-" * 36 + "\n"

    for k in ['phi', 'mu', 'alpha']:
        final = params_final[k]
        target = targets[k]
        err = abs(final - target) / target * 100
        if k == 'mu':
            summary += f"{k:<6} {final:>10.2e} {target:>10.2e} {err:>7.1f}%\n"
        else:
            summary += f"{k:<6} {final:>10.2f} {target:>10.2f} {err:>7.1f}%\n"

    ax12.text(0.1, 0.9, summary, transform=ax12.transAxes,
              fontsize=10, fontfamily='monospace', verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.suptitle('Minor Creek PINN: Physics-Based Displacement Calibration',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  MINOR CREEK LANDSLIDE - CORRECTED PINN")
    print("  Based on: Li, Handwerger, Buscarnera (2023)")
    print("=" * 80)

    # Train
    model, history, data, params_init, params_final = train_pinn(
        epochs=10000,
        lr_net=1e-3,
        lr_params=5e-2,  # Higher LR for physics params
        print_every=500
    )

    # Visualize
    print("\n📊 Creating figure...")
    fig = create_figure(model, history, data, params_init, params_final)

    # Save
    fig.savefig('/mnt/user-data/outputs/minor_creek_pinn_fixed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: minor_creek_pinn_fixed.png")

    plt.show()

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)
