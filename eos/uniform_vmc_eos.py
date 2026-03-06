#!/usr/bin/env python3
"""
uniform_vmc_eos.py
JAX VMC for uniform periodic-box system with frozen u2 Jastrow.
Computes equation of state ε(ρ) for MW-shielded polar molecules.

Simplified from tvmc_mw_shielding_gpu_v7.py:
  - Removes all u1/trap terms (uniform system)
  - Adds periodic boundary conditions via minimum-image convention
  - Scans a grid of densities, running VMC at each
  - Outputs ε(ρ), ε_kin(ρ), ε_int(ρ) with error bars

Usage:
  python uniform_vmc_eos.py                    # Full GPU run
  python uniform_vmc_eos.py --cpu_test         # Quick CPU test
  python uniform_vmc_eos.py --n_walkers 16384  # Custom walker count

Output: eos_vmc_results.npz
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from jax import random as jrand
from functools import partial

print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# Type aliases
Float = jnp.float32
Array = jnp.ndarray


# ==============================================================================
# Physical constants
# ==============================================================================

# Default values (can be overridden via CLI --C3, --C6)
C3 = 0.001
C6 = 1e-6
W1_W2 = 2.0
W0_W2 = 0.0
EPS_R = 0.001
KAPPA_CUSP = 0.1
EPS_CUSP = 0.005
RC_CUSP = 0.02
S_EPS = 1e-10
N2 = 10
ELL2_MIN = 0.01
ELL2_MAX = 0.7


# ==============================================================================
# Frozen u2 parameters
# ==============================================================================

# Basis length scales (log-spaced, matching tvmc_mw_shielding_gpu_v7.py)
ELL2 = jnp.exp(jnp.linspace(
    jnp.log(ELL2_MIN), jnp.log(ELL2_MAX), N2
)).astype(Float)

# Load frozen u2 coefficients from ground-state results
# Layout: params_gs = [a_rho, a_z, c_rho(6), c_z(6), c0(10), c2(10)]
# u2 params are indices 14:34
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
_RESULTS_PATH = os.path.join(
    _SCRIPT_DIR, 'tvmc_output_N8_freeze_u2', 'results.npz'
)
try:
    _data = np.load(_RESULTS_PATH)
    _pgs = _data['params_gs']
    C0 = jnp.array(_pgs[14:24].real, dtype=Float)
    C2 = jnp.array(_pgs[24:34].real, dtype=Float)
    print(f"Loaded frozen u2 from {_RESULTS_PATH}")
except FileNotFoundError:
    print("Warning: results.npz not found, using hardcoded fallback")
    C0 = jnp.array([
        8.9305055e-01, 3.5220504e+00, 1.1713123e-01, -3.7859288e-01,
        -9.1917045e-02, -1.0777023e-01, -3.0403359e-02, -4.9304616e-02,
        3.3019243e-03, -3.6738373e-02
    ], dtype=Float)
    C2 = jnp.array([
        2.9342882e-03, 2.0144200e-02, 5.9530456e-02, 8.8652126e-02,
        3.0827662e-02, -8.6293317e-02, -2.4167843e-02, -6.1543286e-03,
        -7.8385812e-04, -9.1717945e-04
    ], dtype=Float)


# ==============================================================================
# Basis functions (from tvmc_mw_shielding_gpu_v7.py L159-207)
# ==============================================================================

@jit
def g0_vec(r: Array, ell: Array) -> Array:
    """Gaussian RBF: exp(-r²/(2ℓ²))"""
    x = r[..., None] / ell
    return jnp.exp(-0.5 * x * x)


@jit
def g0_dr_vec(r: Array, ell: Array) -> Array:
    """d/dr Gaussian RBF"""
    inv = 1.0 / (ell * ell)
    x = r[..., None] / ell
    return jnp.exp(-0.5 * x * x) * (-r[..., None] * inv)


@jit
def g0_d2r_vec(r: Array, ell: Array) -> Array:
    """d²/dr² Gaussian RBF"""
    inv = 1.0 / (ell * ell)
    r2 = r[..., None] ** 2
    x = r[..., None] / ell
    return jnp.exp(-0.5 * x * x) * (r2 * inv * inv - inv)


@jit
def g2_reg_vec(r: Array, ell: Array) -> Array:
    """Regular l=2 basis: r² * exp(-r²/(2ℓ²))"""
    r2 = r[..., None] ** 2
    x = r[..., None] / ell
    return r2 * jnp.exp(-0.5 * x * x)


@jit
def g2_reg_dr_vec(r: Array, ell: Array) -> Array:
    """d/dr [r² * exp(-r²/(2ℓ²))]"""
    inv = 1.0 / (ell * ell)
    r_exp = r[..., None]
    x = r_exp / ell
    return r_exp * jnp.exp(-0.5 * x * x) * (2.0 - r_exp * r_exp * inv)


@jit
def g2_reg_d2r_vec(r: Array, ell: Array) -> Array:
    """d²/dr² [r² * exp(-r²/(2ℓ²))]"""
    inv = 1.0 / (ell * ell)
    inv2 = inv * inv
    r2 = r[..., None] ** 2
    r4 = r2 * r2
    x = r[..., None] / ell
    return jnp.exp(-0.5 * x * x) * (2.0 - 5.0 * r2 * inv + r4 * inv2)


# ==============================================================================
# Cusp functions (from tvmc_mw_shielding_gpu_v7.py L215-248)
# ==============================================================================

@jit
def cusp_u_vec(r: Array, kappa: float, eps_cusp: float, rc_cusp: float) -> Array:
    """Gaussian-damped cusp: -κ / sqrt(r² + ε²) * exp(-r²/(2*rc²))"""
    r2 = r * r
    s = jnp.sqrt(r2 + eps_cusp * eps_cusp)
    gauss = jnp.exp(-r2 / (2.0 * rc_cusp * rc_cusp))
    return -kappa * gauss / s


@jit
def cusp_du_vec(r: Array, kappa: float, eps_cusp: float, rc_cusp: float) -> Array:
    """First derivative of Gaussian-damped cusp."""
    r2 = r * r
    rc2 = rc_cusp * rc_cusp
    e2 = eps_cusp * eps_cusp
    s2 = r2 + e2
    s = jnp.sqrt(s2)
    gauss = jnp.exp(-r2 / (2.0 * rc2))
    return kappa * gauss * r * (1.0 / rc2 + 1.0 / s2) / s


@jit
def cusp_d2u_vec(r: Array, kappa: float, eps_cusp: float, rc_cusp: float) -> Array:
    """Second derivative of Gaussian-damped cusp."""
    r2 = r * r
    rc2 = rc_cusp * rc_cusp
    s2 = r2 + eps_cusp * eps_cusp
    s = jnp.sqrt(s2)
    s3 = s * s2
    s5 = s2 * s2 * s
    gauss = jnp.exp(-r2 / (2.0 * rc2))
    term1 = (1.0 - r2 / rc2) * (1.0 / rc2 + 1.0 / s2) / s
    term2 = -2.0 * r2 / s5
    term3 = -r2 * (1.0 / rc2 + 1.0 / s2) / s3
    return kappa * gauss * (term1 + term2 + term3)


# ==============================================================================
# Potentials and Jastrows (from tvmc_mw_shielding_gpu_v7.py L256-365)
# ==============================================================================

@jit
def P2(c: Array) -> Array:
    """Legendre polynomial P₂(c) = (3c² - 1)/2"""
    return 0.5 * (3.0 * c * c - 1.0)


@jit
def V_mw_shielding_vec(dx: Array, dy: Array, dz: Array,
                        C3_: float, C6_: float, w1_w2: float, w0_w2: float,
                        eps_r: float) -> Array:
    """Vectorized MW-shielding potential."""
    r2 = dx*dx + dy*dy + dz*dz
    r2_reg = r2 + eps_r * eps_r
    r = jnp.sqrt(r2_reg)
    c = dz / r
    c2 = c * c
    s2 = 1.0 - c2
    s4 = s2 * s2
    s2c2 = s2 * c2
    p2_sq = (3.0 * c2 - 1.0) ** 2
    inv_r6 = 1.0 / (r2_reg * r2_reg * r2_reg)
    V_shield = C6_ * inv_r6 * (s4 + w1_w2 * s2c2 + w0_w2 * p2_sq)
    inv_r3 = 1.0 / (r * r2_reg)
    V_dip = C3_ * inv_r3 * (3.0 * c2 - 1.0)
    return V_shield + V_dip


@jit
def u2_f0_f2_vec(r: Array, c0: Array, c2: Array, ell2: Array,
                  kappa_cusp: float, eps_cusp: float, rc_cusp: float) -> tuple:
    """Compute f0 and f2 values"""
    f0 = cusp_u_vec(r, kappa_cusp, eps_cusp, rc_cusp)
    rbf_vals = g0_vec(r, ell2)
    f0 = f0 + jnp.sum(c0 * rbf_vals, axis=-1)
    g2_vals = g2_reg_vec(r, ell2)
    f2 = jnp.sum(c2 * g2_vals, axis=-1)
    return f0, f2


@jit
def u2_f0_f2_derivs_vec(r: Array, c0: Array, c2: Array, ell2: Array,
                         kappa_cusp: float, eps_cusp: float, rc_cusp: float
                         ) -> tuple:
    """Compute f0, f0', f0'', f2, f2', f2''"""
    f0 = cusp_u_vec(r, kappa_cusp, eps_cusp, rc_cusp)
    f0p = cusp_du_vec(r, kappa_cusp, eps_cusp, rc_cusp)
    f0pp = cusp_d2u_vec(r, kappa_cusp, eps_cusp, rc_cusp)
    rbf_vals = g0_vec(r, ell2)
    rbf_dr = g0_dr_vec(r, ell2)
    rbf_d2r = g0_d2r_vec(r, ell2)
    f0 = f0 + jnp.sum(c0 * rbf_vals, axis=-1)
    f0p = f0p + jnp.sum(c0 * rbf_dr, axis=-1)
    f0pp = f0pp + jnp.sum(c0 * rbf_d2r, axis=-1)
    g2_vals = g2_reg_vec(r, ell2)
    g2_dr = g2_reg_dr_vec(r, ell2)
    g2_d2r = g2_reg_d2r_vec(r, ell2)
    f2 = jnp.sum(c2 * g2_vals, axis=-1)
    f2p = jnp.sum(c2 * g2_dr, axis=-1)
    f2pp = jnp.sum(c2 * g2_d2r, axis=-1)
    return f0, f0p, f0pp, f2, f2p, f2pp


@jit
def u2_val_vec(r: Array, cos_theta: Array, c0: Array, c2: Array, ell2: Array,
               kappa_cusp: float, eps_cusp: float, rc_cusp: float) -> Array:
    """Evaluate u2 = f0(r) + f2(r)*P2(cosθ)"""
    f0, f2 = u2_f0_f2_vec(r, c0, c2, ell2, kappa_cusp, eps_cusp, rc_cusp)
    p2 = P2(cos_theta)
    return f0 + f2 * p2


# ==============================================================================
# Periodic boundary conditions
# ==============================================================================

@jit
def minimum_image(diff: Array, L: float) -> Array:
    """Minimum image convention: wrap displacements into [-L/2, L/2]."""
    return diff - L * jnp.round(diff / L)


# ==============================================================================
# Metropolis sampling with PBC
# ==============================================================================

@partial(jit, static_argnums=(1,))
def metropolis_sweep_batch(X: Array, N: int, L: float,
                           displacements: Array, u_rands: Array) -> tuple:
    """Batched single-particle sweeps with PBC.

    Args:
        X: Walker positions (W, N, 3)
        N: Number of particles (static)
        L: Box size
        displacements: (W, N, 3) proposed moves
        u_rands: (W, N) uniform random numbers for acceptance

    Returns:
        X_new: Updated positions (W, N, 3)
        n_accs: Per-walker acceptance counts (W,)
    """
    def single_walker_sweep(pos, disp, u):
        def body_fn(i, carry):
            pos, n_acc = carry
            old_pos_i = pos[i]
            new_pos_i = old_pos_i + disp[i]
            # PBC wrap to [0, L)
            new_pos_i = new_pos_i - L * jnp.floor(new_pos_i / L)

            # Compute dU = sum_{j!=i} [u2_new(r_ij) - u2_old(r_ij)]
            diff_old = minimum_image(old_pos_i - pos, L)
            diff_new = minimum_image(new_pos_i - pos, L)

            r_old = jnp.sqrt(jnp.sum(diff_old ** 2, axis=-1) + S_EPS ** 2)
            r_new = jnp.sqrt(jnp.sum(diff_new ** 2, axis=-1) + S_EPS ** 2)
            ct_old = jnp.clip(diff_old[:, 2] / r_old, -1.0, 1.0)
            ct_new = jnp.clip(diff_new[:, 2] / r_new, -1.0, 1.0)

            u2_old = u2_val_vec(r_old, ct_old, C0, C2, ELL2,
                                KAPPA_CUSP, EPS_CUSP, RC_CUSP)
            u2_new = u2_val_vec(r_new, ct_new, C0, C2, ELL2,
                                KAPPA_CUSP, EPS_CUSP, RC_CUSP)

            # Index-based mask for self-interaction exclusion
            mask = jnp.arange(N) != i
            dU = jnp.sum(jnp.where(mask, u2_new - u2_old, 0.0))

            log_acc = 2.0 * dU
            accept = (log_acc >= 0.0) | (u[i] < jnp.exp(log_acc))

            new_p = jnp.where(accept, new_pos_i, old_pos_i)
            pos = pos.at[i].set(new_p)
            n_acc = n_acc + accept.astype(Float)
            return pos, n_acc

        pos, n_acc = lax.fori_loop(
            0, N, body_fn, (pos, jnp.array(0.0, dtype=Float)))
        return pos, n_acc

    return vmap(single_walker_sweep)(X, displacements, u_rands)


# ==============================================================================
# Local energy computation
# ==============================================================================

@jit
def compute_local_energy_single(pos: Array, L: float) -> tuple:
    """Compute kinetic and interaction energy for one walker.

    Uses the Hermitian kinetic energy estimator:
        E_kin = +1/2 * sum_i |grad_i U|^2

    This is always >= 0 and equals (1/2)<|nabla ln Psi|^2>.
    We use this instead of the local estimator T_local = -1/2(|nabla U|^2 + lap U)
    because the minimum-image convention creates gradient discontinuities at
    d = L/2, making the analytical Laplacian incorrect (it misses delta-function
    contributions at the MI boundary). The Hermitian estimator only needs the
    gradient, which IS computed correctly.

    Returns:
        E_kin: Total kinetic energy (scalar, always >= 0)
        E_int: Total interaction energy (scalar)
    """
    def particle_derivs(r_i, all_pos):
        """Gradient and potential for particle i."""
        diff = minimum_image(r_i - all_pos, L)
        r2 = jnp.sum(diff * diff, axis=-1)
        r = jnp.sqrt(r2 + S_EPS * S_EPS)
        inv_r = 1.0 / r

        dz = diff[:, 2]
        cos_theta = jnp.clip(dz * inv_r, -1.0, 1.0)
        p2 = P2(cos_theta)

        f0, f0p, f0pp, f2, f2p, f2pp = u2_f0_f2_derivs_vec(
            r, C0, C2, ELL2, KAPPA_CUSP, EPS_CUSP, RC_CUSP)

        s2 = r2 + S_EPS * S_EPS
        g_r = (f0p + f2p * p2) * inv_r - 3.0 * f2 * (dz ** 2) / (s2 * s2)
        g_zadd = 3.0 * f2 * dz / s2

        grad_x = diff[:, 0] * g_r
        grad_y = diff[:, 1] * g_r
        grad_z = diff[:, 2] * g_r + g_zadd

        v_mw = V_mw_shielding_vec(
            diff[:, 0], diff[:, 1], diff[:, 2],
            C3, C6, W1_W2, W0_W2, EPS_R)

        # Distance-based mask for self-interaction (r_ii ≈ S_EPS)
        mask = r > 2.0 * S_EPS

        total_grad = jnp.stack([
            jnp.sum(jnp.where(mask, grad_x, 0.0)),
            jnp.sum(jnp.where(mask, grad_y, 0.0)),
            jnp.sum(jnp.where(mask, grad_z, 0.0))
        ])
        total_v = jnp.sum(jnp.where(mask, v_mw, 0.0))

        return total_grad, total_v

    grad_u2, V_int_pp = vmap(
        particle_derivs, in_axes=(0, None))(pos, pos)

    # Hermitian kinetic energy: +1/2 * sum_i |grad_i U|^2  (always >= 0)
    grad_sq = jnp.sum(grad_u2 * grad_u2, axis=1)
    E_kin = 0.5 * jnp.sum(grad_sq)

    # Interaction energy: factor 0.5 for double counting in vmap
    E_int = 0.5 * jnp.sum(V_int_pp)

    return E_kin, E_int


@jit
def compute_local_energy_batch(X: Array, L: float) -> tuple:
    """Compute energies for all walkers. Returns (E_kin, E_int) each shape (W,)."""
    return vmap(compute_local_energy_single, in_axes=(0, None))(X, L)


# ==============================================================================
# MCMC loops with on-the-fly RNG (memory-efficient)
# ==============================================================================

@partial(jit, static_argnums=(2, 3))
def run_equilibration(X: Array, key, n_equil: int, N: int,
                      L: float, step_size: float) -> tuple:
    """Equilibration sweeps with on-the-fly random number generation."""
    def scan_body(carry, _):
        X, key = carry
        key, k1, k2 = jrand.split(key, 3)
        W = X.shape[0]
        disp = jrand.uniform(k1, (W, N, 3), dtype=Float,
                             minval=-0.5 * step_size, maxval=0.5 * step_size)
        u = jrand.uniform(k2, (W, N), dtype=Float)
        X, _ = metropolis_sweep_batch(X, N, L, disp, u)
        return (X, key), None

    (X, key), _ = lax.scan(scan_body, (X, key), None, length=n_equil)
    return X, key


@partial(jit, static_argnums=(2, 3, 4))
def run_measurements(X: Array, key, n_meas: int, thin: int, N: int,
                     L: float, step_size: float) -> tuple:
    """Measurement loop returning per-batch energy statistics.

    Returns:
        X: Final walker positions
        key: Updated RNG key
        E_kins: Per-batch mean kinetic energies (n_meas,)
        E_ints: Per-batch mean interaction energies (n_meas,)
    """
    def meas_body(carry, _):
        X, key = carry

        # Thinning sweeps
        def thin_step(carry, _):
            X, key = carry
            key, k1, k2 = jrand.split(key, 3)
            W = X.shape[0]
            disp = jrand.uniform(k1, (W, N, 3), dtype=Float,
                                 minval=-0.5 * step_size,
                                 maxval=0.5 * step_size)
            u = jrand.uniform(k2, (W, N), dtype=Float)
            X, _ = metropolis_sweep_batch(X, N, L, disp, u)
            return (X, key), None

        (X, key), _ = lax.scan(thin_step, (X, key), None, length=thin)

        # Compute energies (mean over walkers)
        E_kin_all, E_int_all = compute_local_energy_batch(X, L)
        e_kin = jnp.mean(E_kin_all)
        e_int = jnp.mean(E_int_all)

        return (X, key), (e_kin, e_int)

    (X, key), (E_kins, E_ints) = lax.scan(
        meas_body, (X, key), None, length=n_meas)
    return X, key, E_kins, E_ints


# ==============================================================================
# Step size tuning
# ==============================================================================

def auto_tune_step_size(X, key, N, L, target_rate=0.5, n_iter=20, n_test=5):
    """Binary search for step size giving ~target acceptance rate."""
    step_lo = 0.01 * L
    step_hi = 0.8 * L
    W = X.shape[0]
    best_step = 0.3 * L
    best_diff = 1.0

    for it in range(n_iter):
        step = 0.5 * (step_lo + step_hi)
        total_acc = 0.0
        total_moves = 0

        key_test = key
        X_test = X
        for _ in range(n_test):
            key_test, k1, k2 = jrand.split(key_test, 3)
            disp = jrand.uniform(k1, (W, N, 3), dtype=Float,
                                 minval=-0.5 * step, maxval=0.5 * step)
            u = jrand.uniform(k2, (W, N), dtype=Float)
            X_test, n_accs = metropolis_sweep_batch(X_test, N, L, disp, u)
            total_acc += float(jnp.sum(n_accs))
            total_moves += W * N

        rate = total_acc / total_moves
        diff = abs(rate - target_rate)
        if diff < best_diff:
            best_diff = diff
            best_step = step

        if rate > target_rate:
            step_lo = step
        else:
            step_hi = step

    # Final rate check with best step
    print(f"  Step size: {best_step:.4f} (L={L:.3f}), acceptance: {rate:.3f}")
    return best_step


# ==============================================================================
# Walker initialization
# ==============================================================================

def init_walkers_lattice(n_walkers, N, L, key):
    """Initialize walkers on a simple cubic lattice with small perturbation."""
    n_side = int(np.ceil(N ** (1.0 / 3.0)))
    coords = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(coords) < N:
                    coords.append([
                        (ix + 0.5) / n_side,
                        (iy + 0.5) / n_side,
                        (iz + 0.5) / n_side
                    ])
    coords = jnp.array(coords, dtype=Float) * L  # (N, 3) in [0, L)

    # Replicate for all walkers with small random perturbation
    key, subkey = jrand.split(key)
    pert = jrand.uniform(subkey, (n_walkers, N, 3), dtype=Float,
                         minval=-0.05 * L / n_side,
                         maxval=0.05 * L / n_side)
    X = coords[None, :, :] + pert  # Broadcasting: (n_walkers, N, 3)
    X = X - L * jnp.floor(X / L)   # Wrap to [0, L)
    return X, key


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Uniform VMC equation of state for MW-shielded molecules')
    parser.add_argument('--n_walkers', type=int, default=8192,
                        help='Number of MCMC walkers (default: 8192)')
    parser.add_argument('--n_equil', type=int, default=200,
                        help='Equilibration sweeps (default: 200)')
    parser.add_argument('--n_meas', type=int, default=500,
                        help='Measurement batches (default: 500)')
    parser.add_argument('--thin', type=int, default=5,
                        help='Thinning sweeps between measurements (default: 5)')
    parser.add_argument('--n_rho', type=int, default=20,
                        help='Number of density points (default: 20)')
    parser.add_argument('--rho_min', type=float, default=0.05,
                        help='Minimum density (default: 0.05)')
    parser.add_argument('--rho_max', type=float, default=3.0,
                        help='Maximum density (default: 3.0)')
    parser.add_argument('--output', type=str, default='eos_vmc_results.npz',
                        help='Output file (default: eos_vmc_results.npz)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--cpu_test', action='store_true',
                        help='Quick CPU test with reduced parameters')
    parser.add_argument('--C3', type=float, default=None,
                        help='Dipole interaction strength (default: 0.001)')
    parser.add_argument('--C6', type=float, default=None,
                        help='van der Waals interaction strength (default: 1e-6)')
    parser.add_argument('--N_part', type=int, default=8,
                        help='Number of particles in periodic box (default: 8)')
    parser.add_argument('--kappa_cusp', type=float, default=None,
                        help='Cusp strength (overrides default)')
    parser.add_argument('--eps_cusp', type=float, default=None,
                        help='Cusp regularization (overrides default)')
    parser.add_argument('--rc_cusp', type=float, default=None,
                        help='Cusp range (overrides default)')
    parser.add_argument('--ell2_min', type=float, default=None,
                        help='Min u2 basis length scale (overrides default)')
    parser.add_argument('--ell2_max', type=float, default=None,
                        help='Max u2 basis length scale (overrides default)')
    args = parser.parse_args()

    # Apply interaction parameter overrides
    global C3, C6, KAPPA_CUSP, EPS_CUSP, RC_CUSP, ELL2
    if args.C3 is not None:
        C3 = args.C3
    if args.C6 is not None:
        C6 = args.C6

    # Apply explicit cusp overrides
    if args.kappa_cusp is not None:
        KAPPA_CUSP = args.kappa_cusp
    if args.eps_cusp is not None:
        EPS_CUSP = args.eps_cusp
    if args.rc_cusp is not None:
        RC_CUSP = args.rc_cusp

    # Apply explicit u2 basis length scale overrides (recompute ELL2)
    if args.ell2_min is not None or args.ell2_max is not None:
        ell2_min = args.ell2_min if args.ell2_min is not None else ELL2_MIN
        ell2_max = args.ell2_max if args.ell2_max is not None else ELL2_MAX
        ELL2 = jnp.exp(jnp.linspace(
            jnp.log(ell2_min), jnp.log(ell2_max), N2
        )).astype(Float)
        print(f"  u2 basis: ELL2=[{ell2_min:.4f}, {ell2_max:.4f}]")

    print(f"  Cusp: kappa={KAPPA_CUSP}, eps={EPS_CUSP}, rc={RC_CUSP}")
    print(f"  C3={C3:.6g}, C6={C6:.6g}")

    if args.cpu_test:
        args.n_walkers = 512
        args.n_equil = 50
        args.n_meas = 50
        args.thin = 2
        args.n_rho = 5
        print("=== CPU test mode: reduced parameters ===")

    N = args.N_part
    rho_grid = np.exp(np.linspace(
        np.log(args.rho_min), np.log(args.rho_max), args.n_rho))

    print(f"\nUniform VMC Equation of State")
    print(f"  N={N}, walkers={args.n_walkers}, equil={args.n_equil}, "
          f"meas={args.n_meas}, thin={args.thin}")
    print(f"  C3={C3:.6g}, C6={C6:.6g}")
    print(f"  Density grid: {args.n_rho} points, "
          f"rho in [{args.rho_min:.3f}, {args.rho_max:.3f}]")
    print(f"  Frozen u2: C0[:3]={np.array(C0[:3])}, C2[:3]={np.array(C2[:3])}")
    print()

    key = jrand.PRNGKey(args.seed)

    # Storage arrays
    all_E_per_N = np.zeros(args.n_rho)
    all_E_err = np.zeros(args.n_rho)
    all_Ekin_per_N = np.zeros(args.n_rho)
    all_Ekin_err = np.zeros(args.n_rho)
    all_Eint_per_N = np.zeros(args.n_rho)
    all_Eint_err = np.zeros(args.n_rho)
    all_step_sizes = np.zeros(args.n_rho)
    all_batch_E = np.zeros((args.n_rho, args.n_meas))
    all_batch_Ekin = np.zeros((args.n_rho, args.n_meas))
    all_batch_Eint = np.zeros((args.n_rho, args.n_meas))

    X = None
    L_prev = None
    t_total = time.time()

    for idx, rho in enumerate(rho_grid):
        L = float((N / rho) ** (1.0 / 3.0))
        print(f"--- [{idx+1}/{args.n_rho}] rho = {rho:.4f}, L = {L:.4f} ---")
        t0 = time.time()

        # Initialize or rescale walkers
        if X is None:
            X, key = init_walkers_lattice(args.n_walkers, N, L, key)
            print(f"  Initialized {args.n_walkers} walkers on lattice")
        else:
            # Warm start: rescale positions to new box size
            scale = L / L_prev
            X = X * scale
            X = X - L * jnp.floor(X / L)  # Wrap to [0, L)
            print(f"  Warm start: rescaled from L={L_prev:.3f}")

        # Tune step size for ~50% acceptance
        step_size = auto_tune_step_size(X, key, N, L)
        all_step_sizes[idx] = step_size

        # Equilibration
        print(f"  Equilibrating ({args.n_equil} sweeps)...")
        X, key = run_equilibration(X, key, args.n_equil, N, L, step_size)
        jax.block_until_ready(X)
        t_equil = time.time() - t0
        print(f"  Equilibration: {t_equil:.1f}s")

        # Measurements
        t1 = time.time()
        print(f"  Measuring ({args.n_meas} x {args.thin} sweeps)...")
        X, key, E_kins, E_ints = run_measurements(
            X, key, args.n_meas, args.thin, N, L, step_size)
        jax.block_until_ready(E_kins)
        t_meas = time.time() - t1

        # Convert to numpy and compute statistics
        E_kins_np = np.array(E_kins)
        E_ints_np = np.array(E_ints)
        E_tots_np = E_kins_np + E_ints_np

        n_m = args.n_meas
        e_per_n = np.mean(E_tots_np) / N
        e_err = np.std(E_tots_np, ddof=1) / np.sqrt(n_m) / N
        ekin_per_n = np.mean(E_kins_np) / N
        ekin_err = np.std(E_kins_np, ddof=1) / np.sqrt(n_m) / N
        eint_per_n = np.mean(E_ints_np) / N
        eint_err = np.std(E_ints_np, ddof=1) / np.sqrt(n_m) / N

        all_E_per_N[idx] = e_per_n
        all_E_err[idx] = e_err
        all_Ekin_per_N[idx] = ekin_per_n
        all_Ekin_err[idx] = ekin_err
        all_Eint_per_N[idx] = eint_per_n
        all_Eint_err[idx] = eint_err
        all_batch_E[idx] = E_tots_np / N
        all_batch_Ekin[idx] = E_kins_np / N
        all_batch_Eint[idx] = E_ints_np / N

        print(f"  E/N     = {e_per_n:+.6f} +/- {e_err:.6f}")
        print(f"  E_kin/N = {ekin_per_n:+.6f} +/- {ekin_err:.6f}")
        print(f"  E_int/N = {eint_per_n:+.6f} +/- {eint_err:.6f}")
        print(f"  Measurement: {t_meas:.1f}s, total: {time.time()-t0:.1f}s")

        L_prev = L

        # Save intermediate results (crash recovery)
        np.savez(args.output,
                 rho=rho_grid[:idx+1],
                 E_per_N=all_E_per_N[:idx+1],
                 E_per_N_err=all_E_err[:idx+1],
                 E_kin_per_N=all_Ekin_per_N[:idx+1],
                 E_kin_per_N_err=all_Ekin_err[:idx+1],
                 E_int_per_N=all_Eint_per_N[:idx+1],
                 E_int_per_N_err=all_Eint_err[:idx+1],
                 batch_E=all_batch_E[:idx+1],
                 batch_Ekin=all_batch_Ekin[:idx+1],
                 batch_Eint=all_batch_Eint[:idx+1],
                 step_sizes=all_step_sizes[:idx+1],
                 # Metadata
                 N=N, C3=C3, C6=C6, W1_W2=W1_W2, W0_W2=W0_W2,
                 EPS_R=EPS_R, KAPPA_CUSP=KAPPA_CUSP,
                 EPS_CUSP=EPS_CUSP, RC_CUSP=RC_CUSP,
                 ELL2_min=float(ELL2[0]), ELL2_max=float(ELL2[-1]),
                 n_walkers=args.n_walkers,
                 n_equil=args.n_equil, n_meas=args.n_meas,
                 thin=args.thin, seed=args.seed)

    t_tot = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"Total time: {t_tot:.0f}s ({t_tot/60:.1f} min)")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
