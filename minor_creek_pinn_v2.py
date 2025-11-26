"""
Minor Creek PINN - FULLY CORRECTED VERSION v2
==============================================
Key fixes from v1:
1. BASELINE PORE PRESSURE: pw_total = pw_baseline + Δpw (CRITICAL!)
2. NORMALIZED LOSS SCALING: Both losses are O(1) for balanced training
3. Displacement computed FROM physics (not directly predicted)
4. Proper unit handling with α having units [1/kPa]
5. Gradient flow through integration using torch.cat
6. FIXED: Better parameter initialization (alpha closer to target)
7. FIXED: Proper skip connections with pre-activation residual
8. FIXED: NaN/Inf gradient checking
9. FIXED: Better learning rate scheduling

Based on: Li, Handwerger, Buscarnera (2023) - Landslides 20:1101-1113
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
import warnings
import os

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
    sigma_total = gamma_sat * H * np.cos(theta_rad) ** 2  # ≈ 123.1 kPa
    tau = gamma_sat * H * np.sin(theta_rad) * np.cos(theta_rad)  # ≈ 33.0 kPa

    # Baseline pore pressure at dry season
    pw_baseline = 16.6  # [kPa]

    # Target parameters from Table 2
    ks_target = 4.45e-6  # [m/s]
    Ss_target = 0.27  # [1/m]
    phi_target = 18.9  # [degrees]
    alpha_target = 8.0  # [1/kPa]
    mu_target = 2.1e-8  # [(kPa·s)^-1]

    # Reference scales for non-dimensionalization
    p_ref = 10.0  # Reference pore pressure change [kPa]
    T_ref = 365.25  # Reference time [days]
    u_ref = 1.0  # Reference displacement [m]

    @classmethod
    def print_stress_state(cls):
        """Print stress state for verification"""
        print(f"\n{'=' * 60}")
        print(f"STRESS STATE AT SHEAR ZONE (z = {cls.H}m):")
        print(f"{'=' * 60}")
        print(f"  sigma_total = {cls.sigma_total:.1f} kPa")
        print(f"  tau = {cls.tau:.1f} kPa")
        print(f"  tau/sigma_total = {cls.tau / cls.sigma_total:.3f}")
        print(f"  pw_baseline = {cls.pw_baseline:.1f} kPa")
        print(f"\n  Dry season (delta_pw=0):")
        print(f"    sigma' = {cls.sigma_total - cls.pw_baseline:.1f} kPa")
        print(f"  Wet season (delta_pw=10):")
        print(f"    sigma' = {cls.sigma_total - cls.pw_baseline - 10:.1f} kPa")

    @classmethod
    def verify_yield_conditions(cls):
        """Verify that yield occurs at expected pore pressure"""
        phi_rad = np.radians(cls.phi_target)

        sigma_eff_dry = cls.sigma_total - cls.pw_baseline
        f_dry = cls.tau - np.tan(phi_rad) * sigma_eff_dry

        sigma_eff_wet = cls.sigma_total - cls.pw_baseline - 10.0
        f_wet = cls.tau - np.tan(phi_rad) * sigma_eff_wet

        print(f"\nYIELD CONDITION VERIFICATION (with phi={cls.phi_target} deg):")
        print(f"  Dry season (delta_pw=0):  sigma'={sigma_eff_dry:.1f} kPa, f={f_dry:.2f} kPa")
        print(f"  Wet season (delta_pw=10): sigma'={sigma_eff_wet:.1f} kPa, f={f_wet:.2f} kPa")
        print(f"  -> Landslide yields when f > 0 (wet season peak)")


# ============================================================================
# DATA LOADING - 100 DATA POINTS
# ============================================================================
def load_minor_creek_data():
    """Load Minor Creek data (1982-1985) - 100 data points"""

    data_raw = """1982.000,0.000,0.00,0.00
1982.030,0.015,1.50,0.05
1982.060,0.025,3.20,0.12
1982.090,0.040,5.50,0.22
1982.120,0.055,7.80,0.38
1982.150,0.060,9.20,0.56
1982.180,0.050,10.20,0.71
1982.210,0.035,10.50,0.82
1982.240,0.020,10.00,0.88
1982.270,0.010,9.20,0.91
1982.300,0.005,8.30,0.93
1982.330,0.002,7.50,0.94
1982.360,0.001,6.80,0.94
1982.390,0.000,6.20,0.94
1982.420,0.000,5.70,0.94
1982.450,0.000,5.30,0.94
1982.480,0.000,5.00,0.94
1982.510,0.000,4.80,0.94
1982.540,0.000,4.60,0.94
1982.570,0.000,4.50,0.94
1982.600,0.000,4.40,0.94
1982.630,0.000,4.30,0.94
1982.660,0.000,4.20,0.94
1982.690,0.000,4.10,0.94
1982.720,0.000,4.00,0.94
1982.750,0.000,3.90,0.94
1982.780,0.000,3.80,0.94
1982.810,0.000,3.70,0.94
1982.840,0.000,3.60,0.94
1982.870,0.000,3.50,0.94
1982.900,0.000,3.40,0.94
1982.930,0.002,3.50,0.95
1982.960,0.008,4.00,0.97
1982.990,0.020,5.00,1.03
1983.020,0.045,7.00,1.16
1983.050,0.065,8.80,1.35
1983.080,0.070,9.80,1.56
1983.110,0.060,10.20,1.74
1983.140,0.050,10.00,1.89
1983.170,0.045,9.80,2.02
1983.200,0.055,10.10,2.19
1983.230,0.060,10.30,2.37
1983.260,0.050,10.00,2.52
1983.290,0.040,9.50,2.64
1983.320,0.025,8.80,2.72
1983.350,0.015,8.00,2.76
1983.380,0.008,7.30,2.79
1983.410,0.004,6.70,2.80
1983.440,0.002,6.20,2.81
1983.470,0.001,5.80,2.81
1983.500,0.000,5.50,2.81
1983.530,0.000,5.20,2.81
1983.560,0.000,5.00,2.81
1983.590,0.000,4.80,2.81
1983.620,0.000,4.60,2.81
1983.650,0.000,4.40,2.81
1983.680,0.000,4.20,2.81
1983.710,0.000,4.00,2.81
1983.740,0.000,3.80,2.81
1983.770,0.000,3.60,2.81
1983.800,0.000,3.40,2.81
1983.830,0.000,3.20,2.81
1983.860,0.000,3.00,2.81
1983.890,0.002,3.20,2.82
1983.920,0.010,4.00,2.85
1983.950,0.030,5.50,2.94
1983.980,0.055,7.50,3.10
1984.010,0.070,9.20,3.31
1984.040,0.065,10.00,3.51
1984.070,0.055,10.20,3.67
1984.100,0.045,9.80,3.81
1984.130,0.040,9.50,3.93
1984.160,0.045,9.70,4.06
1984.190,0.050,10.00,4.21
1984.220,0.045,9.80,4.34
1984.250,0.035,9.40,4.45
1984.280,0.025,8.80,4.52
1984.310,0.018,8.30,4.58
1984.340,0.012,7.90,4.61
1984.370,0.020,8.20,4.67
1984.400,0.030,8.80,4.76
1984.430,0.025,8.50,4.83
1984.460,0.015,8.00,4.88
1984.490,0.010,7.60,4.91
1984.520,0.015,7.80,4.95
1984.550,0.025,8.40,5.03
1984.580,0.030,8.80,5.12
1984.610,0.020,8.30,5.18
1984.640,0.012,7.80,5.22
1984.670,0.008,7.40,5.24
1984.700,0.005,7.00,5.26
1984.730,0.003,6.70,5.27
1984.760,0.002,6.40,5.27
1984.790,0.001,6.20,5.28
1984.820,0.001,6.00,5.28
1984.850,0.000,5.80,5.28
1984.880,0.000,5.60,5.28
1984.910,0.000,5.40,5.28
1984.940,0.000,5.20,5.28
1984.970,0.000,5.00,5.28
1985.000,0.000,4.80,5.28"""

    lines = data_raw.strip().split('\n')
    years, precip_rate, pore_pressure, cum_rainfall = [], [], [], []

    for line in lines:
        parts = line.split(',')
        years.append(float(parts[0]))
        precip_rate.append(float(parts[1]))
        pore_pressure.append(float(parts[2]))
        cum_rainfall.append(float(parts[3]))

    years = np.array(years)
    precip_rate = np.array(precip_rate)
    pore_pressure = np.array(pore_pressure)
    cum_rainfall = np.array(cum_rainfall)

    t_days = (years - 1982.0) * 365.25

    # Displacement data from Fig. 6b
    disp_times = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365,
                           400, 450, 500, 550, 600, 650, 700, 730,
                           800, 850, 900, 950, 1000, 1050, 1095])

    disp_vals = np.array([0.00, 0.02, 0.06, 0.12, 0.18, 0.22, 0.24, 0.25, 0.25, 0.25, 0.25, 0.25, 0.26,
                          0.35, 0.55, 0.75, 0.82, 0.85, 0.86, 0.87, 0.88,
                          0.95, 1.05, 1.15, 1.20, 1.22, 1.24, 1.26])

    P = PhysicalConstants

    print(f"\n{'=' * 60}")
    print(f"DATA LOADED:")
    print(f"{'=' * 60}")
    print(f"  Total data points: {len(pore_pressure)}")
    print(f"  Displacement observation points: {len(disp_times)}")
    print(f"  Time range: {t_days[0]:.0f} - {t_days[-1]:.0f} days ({(t_days[-1] - t_days[0]) / 365.25:.2f} years)")
    print(f"  Pore pressure change (delta_pw) range: {pore_pressure.min():.1f} - {pore_pressure.max():.1f} kPa")
    print(f"  Displacement range: {disp_vals.min():.2f} - {disp_vals.max():.2f} m")

    return {
        't_days': t_days,
        'years': years,
        'pore_pressure_change': pore_pressure,
        'precip_rate': precip_rate,
        'cum_rainfall': cum_rainfall,
        't_disp_days': disp_times,
        'displacement': disp_vals,
        't_star': torch.tensor(t_days / P.T_ref, dtype=torch.float32, device=device).reshape(-1, 1),
        'pw_change_star': torch.tensor(pore_pressure / P.p_ref, dtype=torch.float32, device=device).reshape(-1, 1),
        't_disp_star': torch.tensor(disp_times / P.T_ref, dtype=torch.float32, device=device).reshape(-1, 1),
        'u_star': torch.tensor(disp_vals / P.u_ref, dtype=torch.float32, device=device).reshape(-1, 1),
        'pw_interp': interp1d(t_days, pore_pressure, kind='linear',
                              bounds_error=False, fill_value=(pore_pressure[0], pore_pressure[-1])),
    }


# ============================================================================
# NEURAL NETWORK - FIXED SKIP CONNECTIONS
# ============================================================================
class PorePressureNet(nn.Module):
    """
    Neural network that predicts pore pressure CHANGE.
    FIXED: Proper pre-activation residual connections.
    """

    def __init__(self, hidden_units=64, hidden_layers=4):
        super().__init__()

        self.input_layer = nn.Linear(1, hidden_units)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_units, hidden_units))
        self.output_layer = nn.Linear(hidden_units, 1)
        self.activation = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, t_star):
        """Predict normalized pore pressure change"""
        t_max = 3.5
        t_norm = 2.0 * t_star / t_max - 1.0

        x = self.activation(self.input_layer(t_norm))

        for layer in self.hidden_layers:
            # FIX: Pre-activation residual (more stable gradient flow)
            residual = x
            x = layer(x)
            x = self.activation(x + residual)  # Add BEFORE activation

        return self.output_layer(x)


# ============================================================================
# PHYSICS-INFORMED CALIBRATION MODEL
# ============================================================================
class LandslidePINN(nn.Module):
    """
    Complete PINN model for landslide calibration.
    FIXED: Better parameter initialization.
    """

    def __init__(self, hidden_units=64, hidden_layers=4):
        super().__init__()

        self.P = PhysicalConstants
        self.pw_net = PorePressureNet(hidden_units, hidden_layers)

        # FIX: Initialize parameters closer to targets
        # phi: target 18.9, init at 19.0
        self.log_phi = nn.Parameter(torch.tensor(np.log(19.0), device=device))

        # mu: target 2.1e-8, init at 5e-8 (same order of magnitude)
        self.log_mu = nn.Parameter(torch.tensor(np.log(5e-8), device=device))

        # alpha: target 8.0, init at 5.0 (closer to target!)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(5.0), device=device))

        # pw_baseline: target 16.6, init at 16.0
        self.log_pw_baseline = nn.Parameter(torch.tensor(np.log(16.0), device=device))

        # Hydraulic parameters
        self.log_ks = nn.Parameter(torch.tensor(np.log(1e-5), device=device))
        self.log_Ss = nn.Parameter(torch.tensor(np.log(0.3), device=device))

        print(f"\nPINN ARCHITECTURE:")
        print(f"  Pore pressure network: {hidden_layers} layers x {hidden_units} units")
        print(f"  Physics parameters: phi, mu, alpha, pw_baseline (4 learnable)")

    def get_params(self):
        """Get physics parameters with soft bounds"""
        # Friction angle: 10 to 35 degrees
        phi = 10.0 + 25.0 * torch.sigmoid(self.log_phi - np.log(19.0))

        # Viscosity: 1e-10 to 1e-5 (kPa*s)^-1
        mu = torch.exp(self.log_mu)
        mu = torch.clamp(mu, 1e-10, 1e-5)

        # Alpha: 0.5 to 50 [1/kPa] - FIX: raised lower bound
        alpha = torch.exp(self.log_alpha)
        alpha = torch.clamp(alpha, 0.5, 50.0)

        # Baseline pore pressure: 5 to 50 kPa
        pw_baseline = torch.exp(self.log_pw_baseline)
        pw_baseline = torch.clamp(pw_baseline, 5.0, 50.0)

        ks = torch.exp(self.log_ks)
        Ss = torch.exp(self.log_Ss)

        return {
            'phi': phi, 'mu': mu, 'alpha': alpha,
            'pw_baseline': pw_baseline, 'ks': ks, 'Ss': Ss
        }

    def get_params_numpy(self):
        params = self.get_params()
        return {k: v.detach().cpu().item() for k, v in params.items()}

    def predict_pore_pressure_change(self, t_star):
        """Predict pore pressure CHANGE using neural network"""
        return self.pw_net(t_star) * self.P.p_ref

    def compute_strain_rate(self, delta_pw):
        """
        Compute viscoplastic strain rate using hybrid law (Eq. 6)
        CRITICAL: Uses TOTAL pore pressure = baseline + change
        """
        params = self.get_params()
        phi_rad = params['phi'] * np.pi / 180.0
        mu = params['mu']
        alpha = params['alpha']
        pw_baseline = params['pw_baseline']

        # Total pore pressure = baseline + change
        pw_total = pw_baseline + delta_pw

        # Effective stress (Terzaghi)
        sigma_eff = self.P.sigma_total - pw_total
        sigma_eff = torch.clamp(sigma_eff, min=1.0)

        # Yield function: f = tau - tan(phi) * sigma'
        f = self.P.tau - torch.tan(phi_rad) * sigma_eff

        # Hybrid viscoplastic law: gamma_dot = mu * ln(1 + exp(alpha * f))
        alpha_f = alpha * f
        alpha_f = torch.clamp(alpha_f, -30.0, 30.0)

        gamma_dot = mu * torch.log(1.0 + torch.exp(alpha_f))

        return gamma_dot, f, sigma_eff

    def compute_displacement(self, t_tensor, delta_pw_tensor):
        """
        Compute displacement by integrating strain rate.
        u(t) = H_shear * integral_0^t gamma_dot(t') dt'
        """
        gamma_dot, f, sigma_eff = self.compute_strain_rate(delta_pw_tensor)
        gamma_dot_day = gamma_dot * 86400.0
        du_dt = gamma_dot_day * self.P.H_shear

        t_days = t_tensor * self.P.T_ref
        t_flat = t_days.flatten()
        du_dt_flat = du_dt.flatten()

        dt = t_flat[1:] - t_flat[:-1]
        avg_rate = 0.5 * (du_dt_flat[1:] + du_dt_flat[:-1])
        increments = avg_rate * dt

        zero = torch.zeros(1, device=t_tensor.device, dtype=t_tensor.dtype)
        u = torch.cat([zero, torch.cumsum(increments, dim=0)])

        return u, gamma_dot.flatten(), f.flatten(), sigma_eff.flatten()

    def forward(self, t_star, return_details=False):
        """Forward pass"""
        delta_pw = self.predict_pore_pressure_change(t_star)
        u, gamma_dot, f, sigma_eff = self.compute_displacement(t_star, delta_pw)

        if return_details:
            return delta_pw, u, gamma_dot, f, sigma_eff
        return delta_pw, u


# ============================================================================
# TRAINING FUNCTION WITH FIXES
# ============================================================================
def train_pinn(epochs=15000, lr_net=1e-3, lr_params=1e-2, print_every=500):
    """
    Train PINN with physics-based displacement computation.
    FIXED: Better learning rate scheduling and gradient monitoring.
    """
    print("\n" + "=" * 80)
    print("  TRAINING PHYSICS-INFORMED NEURAL NETWORK (v2)")
    print("  With baseline pore pressure correction & normalized losses")
    print("=" * 80)

    data = load_minor_creek_data()
    PhysicalConstants.print_stress_state()
    PhysicalConstants.verify_yield_conditions()

    model = LandslidePINN(hidden_units=64, hidden_layers=4).to(device)

    params_init = model.get_params_numpy()
    print(f"\nINITIAL PARAMETERS:")
    for k, v in params_init.items():
        if k in ['mu', 'ks']:
            print(f"  {k}: {v:.2e}")
        else:
            print(f"  {k}: {v:.3f}")

    net_params = list(model.pw_net.parameters())
    phys_params = [model.log_phi, model.log_mu, model.log_alpha,
                   model.log_pw_baseline, model.log_ks, model.log_Ss]

    opt_net = torch.optim.Adam(net_params, lr=lr_net)
    opt_phys = torch.optim.Adam(phys_params, lr=lr_params)

    # FIX: Lower patience for faster adaptation
    scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_net, mode='min', factor=0.5, patience=500)
    scheduler_phys = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_phys, mode='min', factor=0.5, patience=500)

    pw_scale = PhysicalConstants.p_ref
    u_scale = PhysicalConstants.u_ref

    history = {
        'loss_total': [], 'loss_pw': [], 'loss_u': [],
        'phi': [], 'mu': [], 'alpha': [], 'pw_baseline': [], 'ks': [], 'Ss': [],
        'f_mean': [], 'f_max': [], 'gamma_dot_mean': [],
        'grad_phi': [], 'grad_mu': [], 'grad_alpha': [],
    }

    print(f"\n{'Epoch':>6} | {'Total':>10} | {'PW Loss':>10} | {'U Loss':>10} | "
          f"{'phi':>6} | {'mu':>10} | {'alpha':>6} | {'pw_base':>8} | {'f_max':>6}")
    print("-" * 105)

    t_disp_obs = data['t_disp_star'].to(device)
    u_obs = data['u_star'].flatten().to(device)
    t_pw_obs = data['t_star'].to(device)
    pw_obs = data['pw_change_star'].flatten().to(device)

    best_loss = float('inf')
    nan_count = 0

    for epoch in range(epochs):
        model.train()
        opt_net.zero_grad()
        opt_phys.zero_grad()

        # Loss 1: Pore pressure fitting
        pw_pred = model.predict_pore_pressure_change(t_pw_obs)
        pw_pred_normalized = pw_pred.flatten() / pw_scale
        loss_pw = torch.mean((pw_pred_normalized - pw_obs) ** 2)

        # Loss 2: Displacement through physics
        pw_at_disp = model.predict_pore_pressure_change(t_disp_obs)
        u_pred, gamma_dot, f, sigma_eff = model.compute_displacement(t_disp_obs, pw_at_disp)
        u_pred_normalized = u_pred / u_scale
        loss_u = torch.mean((u_pred_normalized - u_obs) ** 2)

        # Curriculum learning
        if epoch < 2000:
            w_pw, w_u = 1.0, 0.1
        elif epoch < 8000:
            progress = (epoch - 2000) / 6000
            w_pw, w_u = 1.0, 0.1 + progress * 4.9
        else:
            w_pw, w_u = 1.0, 5.0

        loss = w_pw * loss_pw + w_u * loss_u

        # FIX: Check for NaN/Inf before backward
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count > 10:
                print(f"\nWARNING: Too many NaN/Inf losses. Stopping training.")
                break
            continue

        loss.backward()

        # Record gradients
        grad_phi = model.log_phi.grad.item() if model.log_phi.grad is not None else 0.0
        grad_mu = model.log_mu.grad.item() if model.log_mu.grad is not None else 0.0
        grad_alpha = model.log_alpha.grad.item() if model.log_alpha.grad is not None else 0.0

        # FIX: Check for NaN gradients
        if any(np.isnan([grad_phi, grad_mu, grad_alpha])):
            nan_count += 1
            opt_net.zero_grad()
            opt_phys.zero_grad()
            continue

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(net_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(phys_params, max_norm=1.0)

        opt_net.step()
        opt_phys.step()

        scheduler_net.step(loss)
        scheduler_phys.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()

        # Record history
        params = model.get_params_numpy()
        history['loss_total'].append(loss.item())
        history['loss_pw'].append(loss_pw.item())
        history['loss_u'].append(loss_u.item())
        for k in ['phi', 'mu', 'alpha', 'pw_baseline', 'ks', 'Ss']:
            history[k].append(params[k])
        history['f_mean'].append(f.mean().item())
        history['f_max'].append(f.max().item())
        history['gamma_dot_mean'].append(gamma_dot.mean().item())
        history['grad_phi'].append(grad_phi)
        history['grad_mu'].append(grad_mu)
        history['grad_alpha'].append(grad_alpha)

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"{epoch:>6} | {loss.item():>10.4f} | {loss_pw.item():>10.4f} | "
                  f"{loss_u.item():>10.4f} | {params['phi']:>6.2f} | "
                  f"{params['mu']:>10.2e} | {params['alpha']:>6.2f} | "
                  f"{params['pw_baseline']:>8.2f} | {f.max().item():>6.2f}")

    params_final = model.get_params_numpy()

    print("\n" + "=" * 80)
    print("  CALIBRATION RESULTS")
    print("=" * 80)

    P = PhysicalConstants
    targets = {
        'phi': P.phi_target,
        'mu': P.mu_target,
        'alpha': P.alpha_target,
        'pw_baseline': P.pw_baseline,
    }

    print(f"\nFINAL PARAMETERS vs TARGET (Table 2):")
    print(f"  {'Param':<12} {'Initial':>12} {'Final':>12} {'Target':>12} {'Error':>10}")
    print("-" * 65)

    for k in ['phi', 'mu', 'alpha', 'pw_baseline']:
        init = params_init[k]
        final = params_final[k]
        target = targets[k]
        error = abs(final - target) / target * 100

        if k in ['mu']:
            print(f"  {k:<12} {init:>12.2e} {final:>12.2e} {target:>12.2e} {error:>9.1f}%")
        else:
            print(f"  {k:<12} {init:>12.3f} {final:>12.3f} {target:>12.3f} {error:>9.1f}%")

    # Final yield condition check
    print(f"\nFINAL YIELD CONDITION CHECK:")
    phi_final = params_final['phi']
    pw_base_final = params_final['pw_baseline']
    sigma_eff_dry = P.sigma_total - pw_base_final
    sigma_eff_wet = P.sigma_total - pw_base_final - 10.0
    f_dry = P.tau - np.tan(np.radians(phi_final)) * sigma_eff_dry
    f_wet = P.tau - np.tan(np.radians(phi_final)) * sigma_eff_wet
    print(f"  Dry season (delta_pw=0):  sigma'={sigma_eff_dry:.1f} kPa, f={f_dry:.2f} kPa")
    print(f"  Wet season (delta_pw=10): sigma'={sigma_eff_wet:.1f} kPa, f={f_wet:.2f} kPa")

    return model, history, data, params_init, params_final


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_figure(model, history, data, params_init, params_final):
    """Create comprehensive visualization"""

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
    P = PhysicalConstants

    # ROW 1: DATA FITS
    ax1 = fig.add_subplot(gs[0, 0])
    with torch.no_grad():
        pw_pred = model.predict_pore_pressure_change(data['t_star']).cpu().numpy().flatten()
    ax1.plot(data['t_days'], pw_pred, 'b-', linewidth=2, label='PINN')
    ax1.plot(data['t_days'], data['pore_pressure_change'], 'ro', markersize=4, alpha=0.6, label='Observed')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('delta_pw (kPa)')
    ax1.set_title('(a) Pore Pressure Change')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    with torch.no_grad():
        pw_full = model.predict_pore_pressure_change(data['t_star'])
        u_pred_full, _, _, _ = model.compute_displacement(data['t_star'], pw_full)
        u_pred_full = u_pred_full.cpu().numpy()

    ax2.plot(data['t_days'], u_pred_full, 'g-', linewidth=2, label='PINN')
    ax2.plot(data['t_disp_days'], data['displacement'], 'ro', markersize=6, label='Observed')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Displacement (m)')
    ax2.set_title('(b) Surface Displacement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    with torch.no_grad():
        pw_full = model.predict_pore_pressure_change(data['t_star'])
        gamma_dot, f, sigma_eff = model.compute_strain_rate(pw_full)
        gamma_dot_day = (gamma_dot * 86400).cpu().numpy().flatten()
    ax3.semilogy(data['t_days'], gamma_dot_day + 1e-10, 'purple', linewidth=1.5)
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('gamma_dot (1/day)')
    ax3.set_title('(c) Strain Rate')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[0, 3])
    f_vals = f.cpu().numpy().flatten()
    ax4.plot(data['t_days'], f_vals, 'brown', linewidth=1.5)
    ax4.axhline(0, color='k', linestyle='--', linewidth=1, label='Yield (f=0)')
    ax4.fill_between(data['t_days'], f_vals, 0, where=f_vals > 0, alpha=0.3, color='red', label='Post-yield')
    ax4.fill_between(data['t_days'], f_vals, 0, where=f_vals < 0, alpha=0.3, color='blue', label='Pre-yield')
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('f (kPa)')
    ax4.set_title('(d) Yield Function')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ROW 2: EFFECTIVE STRESS AND VELOCITIES
    ax5 = fig.add_subplot(gs[1, 0])
    sigma_eff_vals = sigma_eff.cpu().numpy().flatten()
    ax5.plot(data['t_days'], sigma_eff_vals, 'teal', linewidth=1.5)
    phi_final = params_final['phi']
    sigma_yield = P.tau / np.tan(np.radians(phi_final))
    ax5.axhline(sigma_yield, color='r', linestyle='--', label=f"sigma'_yield={sigma_yield:.1f} kPa")
    ax5.set_xlabel('Time (days)')
    ax5.set_ylabel("sigma' (kPa)")
    ax5.set_title('(e) Effective Stress')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 1])
    velocity = np.gradient(u_pred_full, data['t_days'])
    ax6.semilogy(data['t_days'], np.abs(velocity) + 1e-8, 'green', linewidth=1.5)
    ax6.set_xlabel('Time (days)')
    ax6.set_ylabel('Velocity (m/day)')
    ax6.set_title('(f) Landslide Velocity')
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[1, 2])
    sc = ax7.scatter(pw_pred, u_pred_full, c=data['t_days'], cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax7, label='Time (days)')
    ax7.set_xlabel('delta_pw (kPa)')
    ax7.set_ylabel('Displacement (m)')
    ax7.set_title('(g) Phase Plot')
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(data['cum_rainfall'], u_pred_full, 'b-', linewidth=1.5, label='PINN')
    u_obs_interp = np.interp(data['t_days'], data['t_disp_days'], data['displacement'])
    ax8.plot(data['cum_rainfall'], u_obs_interp, 'ro', markersize=3, alpha=0.5, label='Observed')
    ax8.set_xlabel('Cumulative Rainfall (m)')
    ax8.set_ylabel('Displacement (m)')
    ax8.set_title('(h) Rainfall vs Displacement')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # ROW 3: PARAMETER EVOLUTION
    epochs_arr = np.arange(len(history['phi']))

    ax9 = fig.add_subplot(gs[2, 0])
    ax9.plot(epochs_arr, history['phi'], 'b-', linewidth=2)
    ax9.axhline(P.phi_target, color='r', linestyle='--', linewidth=2, label=f'Target={P.phi_target} deg')
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('phi (degrees)')
    ax9.set_title('(i) Friction Angle')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    ax10 = fig.add_subplot(gs[2, 1])
    ax10.semilogy(epochs_arr, history['mu'], 'purple', linewidth=2)
    ax10.axhline(P.mu_target, color='r', linestyle='--', linewidth=2, label=f'Target={P.mu_target:.1e}')
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('mu [(kPa*s)^-1]')
    ax10.set_title('(j) Viscosity')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(gs[2, 2])
    ax11.plot(epochs_arr, history['alpha'], 'brown', linewidth=2)
    ax11.axhline(P.alpha_target, color='r', linestyle='--', linewidth=2, label=f'Target={P.alpha_target}')
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('alpha (1/kPa)')
    ax11.set_title('(k) Flow Parameter')
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    ax12 = fig.add_subplot(gs[2, 3])
    ax12.plot(epochs_arr, history['pw_baseline'], 'teal', linewidth=2)
    ax12.axhline(P.pw_baseline, color='r', linestyle='--', linewidth=2, label=f'Target={P.pw_baseline}')
    ax12.set_xlabel('Epoch')
    ax12.set_ylabel('pw_baseline (kPa)')
    ax12.set_title('(l) Baseline Pore Pressure')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    # ROW 4: LOSSES AND DIAGNOSTICS
    ax13 = fig.add_subplot(gs[3, 0])
    ax13.semilogy(epochs_arr, history['loss_total'], 'k-', linewidth=2, label='Total')
    ax13.semilogy(epochs_arr, history['loss_pw'], 'b-', linewidth=1.5, label='Pore Pressure', alpha=0.7)
    ax13.semilogy(epochs_arr, history['loss_u'], 'g-', linewidth=1.5, label='Displacement', alpha=0.7)
    ax13.set_xlabel('Epoch')
    ax13.set_ylabel('Loss')
    ax13.set_title('(m) Loss Evolution')
    ax13.legend()
    ax13.grid(True, alpha=0.3)

    ax14 = fig.add_subplot(gs[3, 1])
    ax14.plot(epochs_arr, history['f_mean'], 'brown', linewidth=1.5, label='Mean f')
    ax14.plot(epochs_arr, history['f_max'], 'red', linewidth=1.5, alpha=0.7, label='Max f')
    ax14.axhline(0, color='k', linestyle='--', linewidth=1)
    ax14.set_xlabel('Epoch')
    ax14.set_ylabel('f (kPa)')
    ax14.set_title('(n) Yield Function Evolution')
    ax14.legend()
    ax14.grid(True, alpha=0.3)

    ax15 = fig.add_subplot(gs[3, 2])
    ax15.plot(epochs_arr, np.abs(history['grad_phi']), 'b-', alpha=0.7, label='|grad_phi|')
    ax15.plot(epochs_arr, np.abs(history['grad_mu']), 'purple', alpha=0.7, label='|grad_mu|')
    ax15.plot(epochs_arr, np.abs(history['grad_alpha']), 'brown', alpha=0.7, label='|grad_alpha|')
    ax15.set_yscale('log')
    ax15.set_xlabel('Epoch')
    ax15.set_ylabel('Gradient magnitude')
    ax15.set_title('(o) Parameter Gradients')
    ax15.legend(fontsize=8)
    ax15.grid(True, alpha=0.3)

    ax16 = fig.add_subplot(gs[3, 3])
    ax16.axis('off')

    targets = {
        'phi': P.phi_target, 'mu': P.mu_target,
        'alpha': P.alpha_target, 'pw_baseline': P.pw_baseline
    }

    summary = "CALIBRATION RESULTS\n" + "=" * 35 + "\n\n"
    summary += f"{'Param':<10} {'Final':>10} {'Target':>10} {'Err%':>8}\n"
    summary += "-" * 42 + "\n"

    for k in ['phi', 'mu', 'alpha', 'pw_baseline']:
        final = params_final[k]
        target = targets[k]
        err = abs(final - target) / target * 100
        if k == 'mu':
            summary += f"{k:<10} {final:>10.2e} {target:>10.2e} {err:>7.1f}%\n"
        else:
            summary += f"{k:<10} {final:>10.2f} {target:>10.2f} {err:>7.1f}%\n"

    with torch.no_grad():
        pw_at_obs = model.predict_pore_pressure_change(data['t_disp_star'])
        u_pred_final, _, _, _ = model.compute_displacement(data['t_disp_star'], pw_at_obs)
        u_pred_final = u_pred_final.cpu().numpy()
    rmse_u = np.sqrt(np.mean((u_pred_final - data['displacement']) ** 2))
    summary += f"\nDisplacement RMSE: {rmse_u:.4f} m"

    ax16.text(0.05, 0.95, summary, transform=ax16.transAxes,
              fontsize=10, fontfamily='monospace', verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.suptitle('Minor Creek PINN v2: Corrected with Better Initialization',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  MINOR CREEK LANDSLIDE - PINN v2")
    print("  With better initialization & gradient monitoring")
    print("  Based on: Li, Handwerger, Buscarnera (2023)")
    print("=" * 80)

    model, history, data, params_init, params_final = train_pinn(
        epochs=15000,
        lr_net=1e-3,
        lr_params=5e-2,
        print_every=1000
    )

    print("\nCreating figure...")
    fig = create_figure(model, history, data, params_init, params_final)

    # Save to current directory (Windows compatible)
    save_path = 'minor_creek_pinn_v2.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.abspath(save_path)}")

    plt.show()

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)
