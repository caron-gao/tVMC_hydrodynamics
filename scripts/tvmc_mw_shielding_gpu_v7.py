#!/usr/bin/env python3
"""
tvmc_mw_shielding_gpu.py
JAX/CUDA implementation of TVMC for microwave-shielded polar molecules

Features:
  - Cylindrical trap (omega_rho, omega_z)
  - MW-shielding potential: V_eff = C6/r^6 * [...] + C3/r^3 * (3cos²θ-1)
  - Anisotropic Jastrow: u2 = f0(r) + f2(r)*P2(cosθ)
  - Gaussian-damped cusp term in f0 for short-range hard-core behavior
  - RK4 integrator for real-time evolution
  - JSON Configuration input

GPU Optimizations:
  - Memory-efficient O(N) space complexity using row-wise vmap
  - Batched random number generation
  - lax.scan for MCMC loops
  - Walker configuration reuse between steps
  - Accurate time tracking with adaptive dt

Dependencies: pip install "jax[cuda12]"
"""

from __future__ import annotations

import time
import json
import argparse
import sys
import os
import shutil
from dataclasses import dataclass
from functools import partial, cached_property
from typing import NamedTuple, Literal

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from jax import random as jrand

print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")


# ==============================================================================
# Type aliases
# ==============================================================================

Array = jnp.ndarray
KeyArray = jrand.PRNGKey
Complex = jnp.complex64
Float = jnp.float32


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass(frozen=True)
class TVMCConfig:
    """Configuration for TVMC simulation."""
    N: int
    omega_ref_rho: float
    omega_ref_z: float
    C3: float
    C6: float
    w1_w2: float
    w0_w2: float
    eps_r: float = 0.01
    kappa_cusp: float = 0.1
    eps_cusp: float = 0.01
    rc_cusp: float = 0.3
    n_rho: int = 6
    n_z: int = 6
    n2: int = 10
    ell_rho_min: float = 0.2
    ell_rho_max: float = 3.0
    ell_z_min: float = 0.2
    ell_z_max: float = 3.0
    ell2_min: float = 0.01
    ell2_max: float = 0.5
    n_walkers: int = 4096
    step_size: float = 0.4
    n_burn: int = 20
    n_burn_continue: int = 2
    n_meas: int = 40
    thin: int = 3
    diag_shift: float = 1e-2
    eig_cutoff: float = 1e-4
    omega_floor_rho: float = 0.2
    omega_floor_z: float = 0.2
    max_param_rms: float = 0.01
    s_eps: float = 1e-10
    
    @property
    def nvar(self) -> int:
        return 2 + self.n_rho + self.n_z + 2 * self.n2
    
    @cached_property
    def ell_rho(self) -> Array:
        return jnp.exp(jnp.linspace(
            jnp.log(self.ell_rho_min), jnp.log(self.ell_rho_max), self.n_rho
        )).astype(Float)
    
    @cached_property
    def ell_z(self) -> Array:
        return jnp.exp(jnp.linspace(
            jnp.log(self.ell_z_min), jnp.log(self.ell_z_max), self.n_z
        )).astype(Float)
    
    @cached_property
    def ell2(self) -> Array:
        return jnp.exp(jnp.linspace(
            jnp.log(self.ell2_min), jnp.log(self.ell2_max), self.n2
        )).astype(Float)


# ==============================================================================
# State container
# ==============================================================================

class TVMCState(NamedTuple):
    X: Array        # (n_walkers, N, 3)
    p: Array        # (nvar,) complex
    key: KeyArray


# ==============================================================================
# Cached parameters container
# ==============================================================================

class CachedParams(NamedTuple):
    a_rho: Complex
    a_z: Complex
    c_rho: Array
    c_z: Array
    c0: Array
    c2: Array


def extract_params(p: Array, n_rho: int, n_z: int, n2: int) -> CachedParams:
    """Extract parameters once for reuse."""
    return CachedParams(
        a_rho=p[0],
        a_z=p[1],
        c_rho=p[2:2+n_rho],
        c_z=p[2+n_rho:2+n_rho+n_z],
        c0=p[2+n_rho+n_z:2+n_rho+n_z+n2],
        c2=p[2+n_rho+n_z+n2:]
    )


# ==============================================================================
# Vectorized basis functions
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
# Cusp functions
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
# Potentials and Jastrows
# ==============================================================================

@jit
def P2(c: Array) -> Array:
    """Legendre polynomial P₂(c) = (3c² - 1)/2"""
    return 0.5 * (3.0 * c * c - 1.0)


@jit
def V_mw_shielding_vec(dx: Array, dy: Array, dz: Array,
                        C3: float, C6: float, w1_w2: float, w0_w2: float,
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
    V_shield = C6 * inv_r6 * (s4 + w1_w2 * s2c2 + w0_w2 * p2_sq)
    inv_r3 = 1.0 / (r * r2_reg)
    V_dip = C3 * inv_r3 * (3.0 * c2 - 1.0)
    return V_shield + V_dip


@jit
def u1_val_vec(rho: Array, z: Array, omega_ref_rho: float, omega_ref_z: float,
               a_rho: Complex, a_z: Complex, c_rho: Array, c_z: Array,
               ell_rho: Array, ell_z: Array) -> Array:
    """Evaluate u1 = u_rho(ρ) + u_z(z)"""
    u = -0.5 * (omega_ref_rho + a_rho) * rho * rho
    rbf_rho = g0_vec(rho, ell_rho)
    u = u + jnp.sum(c_rho * rbf_rho, axis=-1)
    u = u + (-0.5 * (omega_ref_z + a_z) * z * z)
    rbf_z = g0_vec(z, ell_z)
    u = u + jnp.sum(c_z * rbf_z, axis=-1)
    return u


@jit
def u1_rho_dr_d2_vec(rho: Array, omega_ref_rho: float, a_rho: Complex,
                      c_rho: Array, ell_rho: Array) -> tuple[Array, Array]:
    """du/dρ and d²u/dρ² for u_rho part"""
    coeff = omega_ref_rho + a_rho
    du = -coeff * rho
    d2 = -coeff * jnp.ones_like(rho)
    rbf_dr = g0_dr_vec(rho, ell_rho)
    rbf_d2r = g0_d2r_vec(rho, ell_rho)
    du = du + jnp.sum(c_rho * rbf_dr, axis=-1)
    d2 = d2 + jnp.sum(c_rho * rbf_d2r, axis=-1)
    return du, d2


@jit
def u1_z_dz_d2_vec(z: Array, omega_ref_z: float, a_z: Complex,
                    c_z: Array, ell_z: Array) -> tuple[Array, Array]:
    """du/dz and d²u/dz² for u_z part"""
    coeff = omega_ref_z + a_z
    du = -coeff * z
    d2 = -coeff * jnp.ones_like(z)
    rbf_dr = g0_dr_vec(z, ell_z)
    rbf_d2r = g0_d2r_vec(z, ell_z)
    du = du + jnp.sum(c_z * rbf_dr, axis=-1)
    d2 = d2 + jnp.sum(c_z * rbf_d2r, axis=-1)
    return du, d2


@jit
def u2_f0_f2_vec(r: Array, c0: Array, c2: Array, ell2: Array,
                  kappa_cusp: float, eps_cusp: float, rc_cusp: float) -> tuple[Array, Array]:
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
                         ) -> tuple[Array, Array, Array, Array, Array, Array]:
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
# Memory-efficient log-psi computation using row-wise vmap (O(N) space)
# ==============================================================================

@jit
def compute_log_psi_optimized(pos: Array, omega_ref_rho: float, omega_ref_z: float,
                               a_rho: Complex, a_z: Complex, c_rho: Array, c_z: Array,
                               ell_rho: Array, ell_z: Array, c0: Array, c2: Array, ell2: Array,
                               kappa_cusp: float, eps_cusp: float, rc_cusp: float,
                               s_eps: float) -> Complex:
    """Memory-efficient log(psi) computation using row-wise vmap."""
    
    # One-body part (O(N))
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    rho = jnp.sqrt(x*x + y*y)
    
    u1 = -0.5 * (omega_ref_rho + a_rho) * jnp.sum(rho * rho)
    u1 = u1 + jnp.sum(c_rho * jnp.sum(g0_vec(rho, ell_rho), axis=0))
    u1 = u1 + (-0.5 * (omega_ref_z + a_z) * jnp.sum(z * z))
    u1 = u1 + jnp.sum(c_z * jnp.sum(g0_vec(z, ell_z), axis=0))

    # Two-body part (O(N) memory via row-wise vmap)
    def particle_interaction(r_i, all_pos):
        """Compute sum of u2 interactions for particle i with all others."""
        diff = r_i - all_pos  # (N, 3) instead of (N, N, 3)
        
        r2 = jnp.sum(diff * diff, axis=-1)
        r = jnp.sqrt(r2 + s_eps * s_eps)
        
        f0_cusp = cusp_u_vec(r, kappa_cusp, eps_cusp, rc_cusp)
        f0_basis = jnp.sum(c0[None, :] * g0_vec(r, ell2), axis=-1)
        f0 = f0_cusp + f0_basis
        
        inv_r = 1.0 / r
        cos_theta = jnp.clip(diff[:, 2] * inv_r, -1.0, 1.0)
        p2 = P2(cos_theta)
        
        f2 = jnp.sum(c2[None, :] * g2_reg_vec(r, ell2), axis=-1)
        
        val = f0 + f2 * p2
        
        # Mask self-interaction
        mask = r > 2.0 * s_eps
        return jnp.sum(jnp.where(mask, val, 0.0))

    u2_per_particle = vmap(particle_interaction, in_axes=(0, None))(pos, pos)
    u2 = 0.5 * jnp.sum(u2_per_particle)  # Factor 0.5 for double counting
    
    return u1 + u2


# ==============================================================================
# Metropolis sampling - Single-particle moves
# ==============================================================================

@partial(jit, static_argnums=(2,))
def metropolis_step_single_particle(pos: Array, i: int, N: int,
                                     omega_ref_rho: float, omega_ref_z: float,
                                     a_rho: Complex, a_z: Complex, c_rho: Array, c_z: Array,
                                     ell_rho: Array, ell_z: Array, c0: Array, c2: Array, ell2: Array,
                                     kappa_cusp: float, eps_cusp: float, rc_cusp: float, s_eps: float,
                                     displacement: Array, u_rand: float) -> Array:
    """Move particle i with Metropolis acceptance."""
    old_pos_i = pos[i]
    new_pos_i = old_pos_i + displacement
    old_rho = jnp.sqrt(old_pos_i[0]**2 + old_pos_i[1]**2)
    new_rho = jnp.sqrt(new_pos_i[0]**2 + new_pos_i[1]**2)
    old_z = old_pos_i[2]
    new_z = new_pos_i[2]
    
    dU = u1_val_vec(new_rho, new_z, omega_ref_rho, omega_ref_z, a_rho, a_z,
                    c_rho, c_z, ell_rho, ell_z) - \
         u1_val_vec(old_rho, old_z, omega_ref_rho, omega_ref_z, a_rho, a_z,
                    c_rho, c_z, ell_rho, ell_z)
    
    diff_old = old_pos_i - pos
    diff_new = new_pos_i - pos
    r_old = jnp.linalg.norm(diff_old, axis=1)
    r_new = jnp.linalg.norm(diff_new, axis=1)
    r_old_safe = jnp.maximum(r_old, s_eps)
    r_new_safe = jnp.maximum(r_new, s_eps)
    cos_theta_old = jnp.clip(diff_old[:, 2] / r_old_safe, -1.0, 1.0)
    cos_theta_new = jnp.clip(diff_new[:, 2] / r_new_safe, -1.0, 1.0)
    
    u2_old = u2_val_vec(r_old, cos_theta_old, c0, c2, ell2, kappa_cusp, eps_cusp, rc_cusp)
    u2_new = u2_val_vec(r_new, cos_theta_new, c0, c2, ell2, kappa_cusp, eps_cusp, rc_cusp)
    
    mask = jnp.arange(N) != i
    dU = dU + jnp.sum(jnp.where(mask, u2_new - u2_old, 0.0 + 0.0j))
    
    log_acc = 2.0 * dU.real
    accept = (log_acc >= 0.0) | (u_rand < jnp.exp(log_acc))
    
    return pos.at[i].set(jnp.where(accept, new_pos_i, old_pos_i))


@partial(jit, static_argnums=(1,))
def metropolis_single_particle_sweep_batch(X: Array, N: int, displacements: Array, u_rands: Array,
                                            omega_ref_rho: float, omega_ref_z: float,
                                            a_rho: Complex, a_z: Complex, c_rho: Array, c_z: Array,
                                            ell_rho: Array, ell_z: Array, c0: Array, c2: Array, ell2: Array,
                                            kappa_cusp: float, eps_cusp: float, rc_cusp: float,
                                            s_eps: float) -> Array:
    """Batched single-particle sweeps over all walkers."""
    def single_walker_sweep(pos, disp, u):
        def body_fn(i, p):
            return metropolis_step_single_particle(
                p, i, N, omega_ref_rho, omega_ref_z, a_rho, a_z,
                c_rho, c_z, ell_rho, ell_z, c0, c2, ell2,
                kappa_cusp, eps_cusp, rc_cusp, s_eps, disp[i], u[i]
            )
        return lax.fori_loop(0, N, body_fn, pos)
    
    return vmap(single_walker_sweep)(X, displacements, u_rands)


# ==============================================================================
# Memory-efficient observables computation (O(N) space)
# ==============================================================================

@partial(jit, static_argnums=(1, 2, 3, 4))
def compute_observables_single_optimized(pos: Array, N: int, n_rho: int, n_z: int, n2: int,
                                          omega_H_rho: float, omega_H_z: float,
                                          omega_ref_rho: float, omega_ref_z: float,
                                          C3: float, C6: float, w1_w2: float, w0_w2: float, eps_r: float,
                                          a_rho: Complex, a_z: Complex,
                                          c_rho: Array, c_z: Array, ell_rho: Array, ell_z: Array,
                                          c0: Array, c2: Array, ell2: Array,
                                          kappa_cusp: float, eps_cusp: float, rc_cusp: float,
                                          s_eps: float) -> tuple[Array, Complex, float, float]:
    """Memory-efficient observable computation using row-wise vmap."""
    nvar = 2 + n_rho + n_z + 2 * n2
    
    # One-body quantities
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    rho2 = x*x + y*y
    z2 = z*z
    rho = jnp.sqrt(rho2)
    
    du_rho, d2_rho = u1_rho_dr_d2_vec(rho, omega_ref_rho, a_rho, c_rho, ell_rho)
    du_z, d2_z = u1_z_dz_d2_vec(z, omega_ref_z, a_z, c_z, ell_z)
    
    inv_rho = 1.0 / jnp.maximum(rho, 1e-7)
    grad_u1 = jnp.stack([du_rho * inv_rho * x, du_rho * inv_rho * y, du_z], axis=-1)
    lap_u1 = (d2_rho + du_rho * inv_rho) + d2_z

    # Two-body gradients and energy via row-wise vmap (O(N) memory)
    def particle_derivatives(r_i, all_pos):
        """Compute gradient and laplacian contributions for particle i."""
        diff = r_i - all_pos  # (N, 3)
        r2 = jnp.sum(diff * diff, axis=-1)
        r = jnp.sqrt(r2 + s_eps * s_eps)
        inv_r = 1.0 / r
        
        dz = diff[:, 2]
        cos_theta = jnp.clip(dz * inv_r, -1.0, 1.0)
        p2 = P2(cos_theta)
        
        f0, f0p, f0pp, f2, f2p, f2pp = u2_f0_f2_derivs_vec(
            r, c0, c2, ell2, kappa_cusp, eps_cusp, rc_cusp
        )
        
        s2 = r2 + s_eps * s_eps
        g_r = (f0p + f2p * p2) * inv_r - (3.0 * f2 * (dz ** 2)) / (s2 * s2)
        g_zadd = (3.0 * f2 * dz) / s2
        
        grad_x = diff[:, 0] * g_r
        grad_y = diff[:, 1] * g_r
        grad_z = diff[:, 2] * g_r + g_zadd
        
        lap_val = f0pp + 2.0 * f0p * inv_r + (f2pp + 2.0 * f2p * inv_r - 6.0 * f2 / s2) * p2
        
        v_mw = V_mw_shielding_vec(diff[:, 0], diff[:, 1], diff[:, 2],
                                   C3, C6, w1_w2, w0_w2, eps_r)
        
        mask = r > 2.0 * s_eps
        
        total_grad = jnp.stack([
            jnp.sum(jnp.where(mask, grad_x, 0.0)),
            jnp.sum(jnp.where(mask, grad_y, 0.0)),
            jnp.sum(jnp.where(mask, grad_z, 0.0))
        ])
        total_lap = jnp.sum(jnp.where(mask, lap_val, 0.0))
        total_v = jnp.sum(jnp.where(mask, v_mw, 0.0))
        
        return total_grad, total_lap, total_v

    grad_u2, lap_u2, V_int_per_particle = vmap(
        particle_derivatives, in_axes=(0, None)
    )(pos, pos)
    
    # Total gradient and kinetic energy
    grad_tot = grad_u1 + grad_u2
    grad_sq = jnp.sum(grad_tot * grad_tot, axis=1)
    kin_local = -0.5 * jnp.sum(grad_sq + lap_u1 + lap_u2)
    
    # Potential energies
    Vext = 0.5 * (omega_H_rho ** 2) * jnp.sum(rho2) + 0.5 * (omega_H_z ** 2) * jnp.sum(z2)
    Vint = 0.5 * jnp.sum(V_int_per_particle)
    
    E_loc = kin_local + Vext + Vint
    
    # O_k observables (one-body terms)
    O = jnp.zeros(nvar, dtype=Float)
    O = O.at[0].set(-0.5 * jnp.sum(rho2))
    O = O.at[1].set(-0.5 * jnp.sum(z2))
    O = O.at[2:2+n_rho].set(jnp.sum(g0_vec(rho, ell_rho), axis=0))
    O = O.at[2+n_rho:2+n_rho+n_z].set(jnp.sum(g0_vec(z, ell_z), axis=0))
    
    # O_k two-body terms via row-wise computation
    def particle_ok_two_body(r_i, all_pos):
        diff = r_i - all_pos
        r2 = jnp.sum(diff * diff, axis=-1)
        r = jnp.sqrt(r2 + s_eps * s_eps)
        inv_r = 1.0 / r
        cos_theta = jnp.clip(diff[:, 2] * inv_r, -1.0, 1.0)
        p2 = P2(cos_theta)
        
        mask = r > 2.0 * s_eps
        
        g0_vals = g0_vec(r, ell2)  # (N, n2)
        g2_vals = g2_reg_vec(r, ell2)  # (N, n2)
        
        ok_c0 = jnp.sum(jnp.where(mask[:, None], g0_vals, 0.0), axis=0)
        ok_c2 = jnp.sum(jnp.where(mask[:, None], g2_vals * p2[:, None], 0.0), axis=0)
        
        return ok_c0, ok_c2
    
    ok_c0_per_particle, ok_c2_per_particle = vmap(
        particle_ok_two_body, in_axes=(0, None)
    )(pos, pos)
    
    O = O.at[2+n_rho+n_z:2+n_rho+n_z+n2].set(0.5 * jnp.sum(ok_c0_per_particle, axis=0))
    O = O.at[2+n_rho+n_z+n2:].set(0.5 * jnp.sum(ok_c2_per_particle, axis=0))

    return O, E_loc, jnp.mean(rho2), jnp.mean(z2)


@partial(jit, static_argnums=(1, 2, 3, 4))
def compute_observables_batch(X: Array, N: int, n_rho: int, n_z: int, n2: int,
                               omega_H_rho: float, omega_H_z: float,
                               omega_ref_rho: float, omega_ref_z: float,
                               C3: float, C6: float, w1_w2: float, w0_w2: float, eps_r: float,
                               a_rho: Complex, a_z: Complex,
                               c_rho: Array, c_z: Array, ell_rho: Array, ell_z: Array,
                               c0: Array, c2: Array, ell2: Array,
                               kappa_cusp: float, eps_cusp: float, rc_cusp: float,
                               s_eps: float) -> tuple[Array, Array, Array, Array]:
    """Vectorized observables over all walkers."""
    return vmap(lambda pos: compute_observables_single_optimized(
        pos, N, n_rho, n_z, n2, omega_H_rho, omega_H_z, omega_ref_rho, omega_ref_z,
        C3, C6, w1_w2, w0_w2, eps_r, a_rho, a_z, c_rho, c_z, ell_rho, ell_z,
        c0, c2, ell2, kappa_cusp, eps_cusp, rc_cusp, s_eps
    ))(X)


# ==============================================================================
# Optimized MCMC with batched RNG and lax.scan
# ==============================================================================

@partial(jit, static_argnums=(2, 3, 4))
def run_mcmc_burn_optimized(X: Array, key: KeyArray, n_burn: int, W: int, N: int,
                             step_size: float,
                             omega_ref_rho: float, omega_ref_z: float,
                             a_rho: Complex, a_z: Complex,
                             c_rho: Array, c_z: Array, ell_rho: Array, ell_z: Array,
                             c0: Array, c2: Array, ell2: Array,
                             kappa_cusp: float, eps_cusp: float, rc_cusp: float,
                             s_eps: float) -> tuple[Array, KeyArray]:
    """Optimized burn-in with batched RNG and lax.scan."""
    key, key_disp, key_u = jrand.split(key, 3)
    all_displacements = jrand.uniform(
        key_disp, (n_burn, W, N, 3), dtype=Float,
        minval=-0.5*step_size, maxval=0.5*step_size
    )
    all_u_rands = jrand.uniform(key_u, (n_burn, W, N), dtype=Float)
    
    def scan_body(X, inputs):
        disp, u = inputs
        X_new = metropolis_single_particle_sweep_batch(
            X, N, disp, u,
            omega_ref_rho, omega_ref_z, a_rho, a_z,
            c_rho, c_z, ell_rho, ell_z, c0, c2, ell2,
            kappa_cusp, eps_cusp, rc_cusp, s_eps
        )
        return X_new, None
    
    X_final, _ = lax.scan(scan_body, X, (all_displacements, all_u_rands))
    return X_final, key


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def run_mcmc_measurements_optimized(X: Array, key: KeyArray,
                                     n_meas: int, thin: int, W: int, N: int,
                                     n_rho: int, n_z: int, n2: int,
                                     step_size: float,
                                     omega_H_rho: float, omega_H_z: float,
                                     omega_ref_rho: float, omega_ref_z: float,
                                     C3: float, C6: float, w1_w2: float, w0_w2: float, eps_r: float,
                                     a_rho: Complex, a_z: Complex,
                                     c_rho: Array, c_z: Array, ell_rho: Array, ell_z: Array,
                                     c0: Array, c2: Array, ell2: Array,
                                     kappa_cusp: float, eps_cusp: float, rc_cusp: float,
                                     s_eps: float
                                     ) -> tuple[Array, KeyArray, Array, Complex, Float, Float, Array, Array]:
    """Optimized measurements with batched RNG and lax.scan."""
    nvar = 2 + n_rho + n_z + 2 * n2
    
    total_sweeps = n_meas * thin
    key, key_disp, key_u = jrand.split(key, 3)
    all_displacements = jrand.uniform(
        key_disp, (total_sweeps, W, N, 3), dtype=Float,
        minval=-0.5*step_size, maxval=0.5*step_size
    )
    all_u_rands = jrand.uniform(key_u, (total_sweeps, W, N), dtype=Float)
    
    all_displacements = all_displacements.reshape(n_meas, thin, W, N, 3)
    all_u_rands = all_u_rands.reshape(n_meas, thin, W, N)
    
    def measurement_body(carry, inputs):
        X, sO, sE, sR, sZ, sOO, sOE = carry
        disps_thin, us_thin = inputs
        
        def thin_body(X, thin_inputs):
            disp, u = thin_inputs
            return metropolis_single_particle_sweep_batch(
                X, N, disp, u,
                omega_ref_rho, omega_ref_z, a_rho, a_z,
                c_rho, c_z, ell_rho, ell_z, c0, c2, ell2,
                kappa_cusp, eps_cusp, rc_cusp, s_eps
            ), None
        
        X, _ = lax.scan(thin_body, X, (disps_thin, us_thin))
        
        O, E, R, Z = compute_observables_batch(
            X, N, n_rho, n_z, n2, omega_H_rho, omega_H_z, omega_ref_rho, omega_ref_z,
            C3, C6, w1_w2, w0_w2, eps_r, a_rho, a_z, c_rho, c_z, ell_rho, ell_z,
            c0, c2, ell2, kappa_cusp, eps_cusp, rc_cusp, s_eps
        )
        
        new_sO = sO + jnp.sum(O, axis=0)
        new_sE = sE + jnp.sum(E)
        new_sR = sR + jnp.sum(R)
        new_sZ = sZ + jnp.sum(Z)
        new_sOO = sOO + O.T @ O
        new_sOE = sOE + O.T @ E
        
        return (X, new_sO, new_sE, new_sR, new_sZ, new_sOO, new_sOE), None
    
    init_carry = (
        X,
        jnp.zeros(nvar, dtype=Float),
        jnp.array(0.0 + 0.0j, dtype=Complex),
        jnp.array(0.0, dtype=Float),
        jnp.array(0.0, dtype=Float),
        jnp.zeros((nvar, nvar), dtype=Float),
        jnp.zeros(nvar, dtype=Complex)
    )
    
    (X_final, sO, sE, sR, sZ, sOO, sOE), _ = lax.scan(
        measurement_body, init_carry, (all_displacements, all_u_rands)
    )
    
    return X_final, key, sO, sE, sR, sZ, sOO, sOE


def run_mcmc_and_estimate(X: Array, key: KeyArray, cfg: TVMCConfig,
                          omega_H_rho: float, omega_H_z: float,
                          p: Array, n_burn: int) -> tuple[Array, Array, Array, Complex, float, float, KeyArray]:
    """Full MCMC estimation."""
    params = extract_params(p, cfg.n_rho, cfg.n_z, cfg.n2)
    
    X, key = run_mcmc_burn_optimized(
        X, key, n_burn, cfg.n_walkers, cfg.N, cfg.step_size,
        cfg.omega_ref_rho, cfg.omega_ref_z, params.a_rho, params.a_z,
        params.c_rho, params.c_z, cfg.ell_rho, cfg.ell_z,
        params.c0, params.c2, cfg.ell2, cfg.kappa_cusp, cfg.eps_cusp, cfg.rc_cusp, cfg.s_eps
    )
    
    X, key, sO, sE, sR, sZ, sOO, sOE = run_mcmc_measurements_optimized(
        X, key, cfg.n_meas, cfg.thin, cfg.n_walkers, cfg.N,
        cfg.n_rho, cfg.n_z, cfg.n2, cfg.step_size,
        omega_H_rho, omega_H_z, cfg.omega_ref_rho, cfg.omega_ref_z,
        cfg.C3, cfg.C6, cfg.w1_w2, cfg.w0_w2, cfg.eps_r,
        params.a_rho, params.a_z, params.c_rho, params.c_z, cfg.ell_rho, cfg.ell_z,
        params.c0, params.c2, cfg.ell2, cfg.kappa_cusp, cfg.eps_cusp, cfg.rc_cusp, cfg.s_eps
    )
    
    count = cfg.n_meas * cfg.n_walkers
    mO, mE = sO / count, sE / count
    S = (sOO / count) - jnp.outer(mO, mO)
    F = (sOE / count) - (mO * mE)
    S = 0.5 * (S + S.T) + cfg.diag_shift * jnp.eye(S.shape[0], dtype=Float)
    
    return X, S, F, mE, sR/count, sZ/count, key


# ==============================================================================
# TDVP Utilities
# ==============================================================================

@jit
def solve_metric_eigh(S: Array, rhs: Array, eig_cutoff: float) -> Array:
    """Eigenvalue-truncated solve."""
    w, v = jnp.linalg.eigh(S)
    return v @ ((v.T @ rhs) * jnp.where(w > eig_cutoff, 1.0 / w, 0.0))


@jit
def stabilize_dt_jit(omega_ref_rho: float, omega_ref_z: float,
                     omega_floor_rho: float, omega_floor_z: float,
                     max_param_rms: float, p: Array, dp: Array, dt: float) -> float:
    """
    Adaptive time-step limiter.
    
    Prevents:
    1. Effective trap frequency from dropping below floor
    2. Total parameter change from exceeding threshold
    """
    # Check rho direction
    omega_eff_rho = omega_ref_rho + p[0].real
    domega_rho = dp[0].real
    headroom_rho = omega_eff_rho - omega_floor_rho
    dt_max_rho = 0.9 * headroom_rho / (-domega_rho + 1e-12)
    dt = lax.cond((domega_rho < 0) & (headroom_rho > 0),
                  lambda: jnp.minimum(dt, dt_max_rho),
                  lambda: lax.cond((domega_rho < 0) & (headroom_rho <= 0),
                                   lambda: 0.0, lambda: dt))
    
    # Check z direction
    omega_eff_z = omega_ref_z + p[1].real
    domega_z = dp[1].real
    headroom_z = omega_eff_z - omega_floor_z
    dt_max_z = 0.9 * headroom_z / (-domega_z + 1e-12)
    dt = lax.cond((domega_z < 0) & (headroom_z > 0),
                  lambda: jnp.minimum(dt, dt_max_z),
                  lambda: lax.cond((domega_z < 0) & (headroom_z <= 0),
                                   lambda: 0.0, lambda: dt))
    
    # Limit total parameter change
    rms = jnp.sqrt(jnp.mean(jnp.abs(dt * dp) ** 2))
    return lax.cond(rms > max_param_rms,
                    lambda: dt * max_param_rms / (rms + 1e-12),
                    lambda: jnp.maximum(dt, 0.0))


@jit
def apply_omega_floor(p_new: Array, omega_ref_rho: float, omega_ref_z: float,
                      omega_floor_rho: float, omega_floor_z: float) -> Array:
    """Apply omega floor constraints."""
    p_new = lax.cond(
        omega_ref_rho + p_new[0].real < omega_floor_rho,
        lambda: p_new.at[0].set((omega_floor_rho - omega_ref_rho) + 1j * p_new[0].imag),
        lambda: p_new
    )
    p_new = lax.cond(
        omega_ref_z + p_new[1].real < omega_floor_z,
        lambda: p_new.at[1].set((omega_floor_z - omega_ref_z) + 1j * p_new[1].imag),
        lambda: p_new
    )
    return p_new


# ==============================================================================
# TDVP Integrators
# ==============================================================================

def tdvp_step_rk4(state: TVMCState, cfg: TVMCConfig,
                  omega_H_rho: float, omega_H_z: float, dt: float,
                  mode: Literal["real", "imag"], first_step: bool = False
                  ) -> tuple[TVMCState, Complex, float, float, float]:
    """RK4 step. Returns (state, E, rho2, z2, dt_actual)."""
    X, p, key = state.X, state.p, state.key
    
    n_burn_k1 = cfg.n_burn if first_step else cfg.n_burn_continue
    n_burn_rest = cfg.n_burn_continue
    
    # k1
    X, S1, F1, E1, rho2_1, z2_1, key = run_mcmc_and_estimate(
        X, key, cfg, omega_H_rho, omega_H_z, p, n_burn_k1
    )
    rhs1 = -1j * F1 if mode == "real" else -F1
    k1 = solve_metric_eigh(S1, rhs1, cfg.eig_cutoff)
    dt1 = stabilize_dt_jit(cfg.omega_ref_rho, cfg.omega_ref_z,
                           cfg.omega_floor_rho, cfg.omega_floor_z,
                           cfg.max_param_rms, p, k1, dt)
    p1 = p + 0.5 * dt1 * k1
    
    # k2
    X, S2, F2, E2, rho2_2, z2_2, key = run_mcmc_and_estimate(
        X, key, cfg, omega_H_rho, omega_H_z, p1, n_burn_rest
    )
    rhs2 = -1j * F2 if mode == "real" else -F2
    k2 = solve_metric_eigh(S2, rhs2, cfg.eig_cutoff)
    dt2 = stabilize_dt_jit(cfg.omega_ref_rho, cfg.omega_ref_z,
                           cfg.omega_floor_rho, cfg.omega_floor_z,
                           cfg.max_param_rms, p, k2, dt1)
    p2 = p + 0.5 * dt2 * k2
    
    # k3
    X, S3, F3, E3, rho2_3, z2_3, key = run_mcmc_and_estimate(
        X, key, cfg, omega_H_rho, omega_H_z, p2, n_burn_rest
    )
    rhs3 = -1j * F3 if mode == "real" else -F3
    k3 = solve_metric_eigh(S3, rhs3, cfg.eig_cutoff)
    dt3 = stabilize_dt_jit(cfg.omega_ref_rho, cfg.omega_ref_z,
                           cfg.omega_floor_rho, cfg.omega_floor_z,
                           cfg.max_param_rms, p, k3, dt2)
    p3 = p + dt3 * k3
    
    # k4
    X, S4, F4, E4, rho2_4, z2_4, key = run_mcmc_and_estimate(
        X, key, cfg, omega_H_rho, omega_H_z, p3, n_burn_rest
    )
    rhs4 = -1j * F4 if mode == "real" else -F4
    k4 = solve_metric_eigh(S4, rhs4, cfg.eig_cutoff)
    
    # RK4 combination
    k_avg = (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
    dt_final = stabilize_dt_jit(cfg.omega_ref_rho, cfg.omega_ref_z,
                                cfg.omega_floor_rho, cfg.omega_floor_z,
                                cfg.max_param_rms, p, k_avg, dt3)
    p_new = p + dt_final * k_avg
    
    p_new = apply_omega_floor(p_new, cfg.omega_ref_rho, cfg.omega_ref_z,
                              cfg.omega_floor_rho, cfg.omega_floor_z)
    
    if mode == "imag":
        p_new = p_new.real.astype(Float) + 0.0j
    
    return TVMCState(X=X, p=p_new, key=key), E4, rho2_4, z2_4, float(dt_final)


def tdvp_step_rk2(state: TVMCState, cfg: TVMCConfig,
                  omega_H_rho: float, omega_H_z: float, dt: float,
                  mode: Literal["real", "imag"], first_step: bool = False
                  ) -> tuple[TVMCState, Complex, float, float, float]:
    """RK2 (midpoint) step. Returns (state, E, rho2, z2, dt_actual)."""
    X, p, key = state.X, state.p, state.key
    
    n_burn_k1 = cfg.n_burn if first_step else cfg.n_burn_continue
    n_burn_k2 = cfg.n_burn_continue
    
    X, S, F, E1, rho2_1, z2_1, key = run_mcmc_and_estimate(
        X, key, cfg, omega_H_rho, omega_H_z, p, n_burn_k1
    )
    rhs1 = -1j * F if mode == "real" else -F
    dp1 = solve_metric_eigh(S, rhs1, cfg.eig_cutoff)
    dt1 = stabilize_dt_jit(cfg.omega_ref_rho, cfg.omega_ref_z,
                           cfg.omega_floor_rho, cfg.omega_floor_z,
                           cfg.max_param_rms, p, dp1, dt)
    p_mid = p + 0.5 * dt1 * dp1
    
    X, S, F, E2, rho2_2, z2_2, key = run_mcmc_and_estimate(
        X, key, cfg, omega_H_rho, omega_H_z, p_mid, n_burn_k2
    )
    rhs2 = -1j * F if mode == "real" else -F
    dp2 = solve_metric_eigh(S, rhs2, cfg.eig_cutoff)
    dt2 = stabilize_dt_jit(cfg.omega_ref_rho, cfg.omega_ref_z,
                           cfg.omega_floor_rho, cfg.omega_floor_z,
                           cfg.max_param_rms, p, dp2, dt1)
    p_new = p + dt2 * dp2
    
    p_new = apply_omega_floor(p_new, cfg.omega_ref_rho, cfg.omega_ref_z,
                              cfg.omega_floor_rho, cfg.omega_floor_z)
    
    if mode == "imag":
        p_new = p_new.real.astype(Float) + 0.0j
    
    return TVMCState(X=X, p=p_new, key=key), E2, rho2_2, z2_2, float(dt2)


# ==============================================================================
# Initialization and utilities
# ==============================================================================

def init_state(cfg: TVMCConfig, key: KeyArray) -> TVMCState:
    """Initialize walkers and parameters."""
    key, k1, k2, k3, k4 = jrand.split(key, 5)
    std_xy = jnp.sqrt(1.0 / (2.0 * cfg.omega_ref_rho))
    std_z = jnp.sqrt(1.0 / (2.0 * cfg.omega_ref_z))
    X = jnp.empty((cfg.n_walkers, cfg.N, 3), dtype=Float)
    X = X.at[:, :, 0].set(jrand.normal(k2, (cfg.n_walkers, cfg.N), dtype=Float) * std_xy)
    X = X.at[:, :, 1].set(jrand.normal(k3, (cfg.n_walkers, cfg.N), dtype=Float) * std_xy)
    X = X.at[:, :, 2].set(jrand.normal(k4, (cfg.n_walkers, cfg.N), dtype=Float) * std_z)
    return TVMCState(X=X, p=jnp.zeros(cfg.nvar, dtype=Complex), key=key)


def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{path}': {e}")
        sys.exit(1)


def setup_output_directory(config_path: str, output_config: dict) -> tuple[str, str, str]:
    """Create output directory and copy config file."""
    base_dir = output_config.get('directory', 'output')
    os.makedirs(base_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(base_dir)}")
    
    try:
        config_filename = os.path.basename(config_path)
        destination_config = os.path.join(base_dir, config_filename)
        shutil.copy(config_path, destination_config)
        print(f"Configuration copied to: {destination_config}")
    except Exception as e:
        print(f"Warning: Could not copy config file: {e}")
    
    results_path = os.path.join(base_dir, output_config.get('results_file', 'results.npz'))
    walkers_path = os.path.join(base_dir, output_config.get('walkers_file', 'walkers.npy'))
    
    return base_dir, results_path, walkers_path


def compute_aspect_ratio(rho2: float, z2: float) -> float:
    """Aspect ratio = sqrt(<x²>/<z²>) = sqrt(<ρ²>/(2<z²>))"""
    return float(jnp.sqrt(rho2 / (2.0 * z2))) if z2 > 1e-12 else float('inf')


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="JAX/CUDA TVMC Simulation")
    parser.add_argument("config", nargs="?", default="config.json",
                        help="Path to JSON configuration file (default: config.json)")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("JAX/CUDA TVMC: MW-Shielding with Memory-Efficient O(N) Space")
    print(f"{'='*70}")
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    c = load_config(args.config)
    
    sys_c = c['system']
    pot_c = c['potential']
    cusp_c = c['cusp']
    bas_c = c['basis']
    mc_c = c['mcmc']
    tdvp_c = c['tdvp']
    evo_c = c['evolution']
    out_c = c.get('output', {'directory': 'output', 'results_file': 'results.npz',
                              'walkers_file': 'walkers.npy'})
    
    output_dir, results_path, walkers_path = setup_output_directory(args.config, out_c)
    
    n_burn_continue = mc_c.get('n_burn_continue', max(2, mc_c['n_burn'] // 5))
    
    cfg = TVMCConfig(
        N=sys_c['N'],
        omega_ref_rho=sys_c['omega_rho'],
        omega_ref_z=sys_c['omega_z'],
        C3=pot_c['C3'], C6=pot_c['C6'], w1_w2=pot_c['w1_w2'],
        w0_w2=pot_c['w0_w2'], eps_r=pot_c['eps_r'],
        kappa_cusp=cusp_c['kappa'], eps_cusp=cusp_c['eps'], rc_cusp=cusp_c['rc'],
        n_rho=bas_c['n_rho'], n_z=bas_c['n_z'], n2=bas_c['n2'],
        ell_rho_min=bas_c['ell_rho_min'], ell_rho_max=bas_c['ell_rho_max'],
        ell_z_min=bas_c['ell_z_min'], ell_z_max=bas_c['ell_z_max'],
        ell2_min=bas_c['ell2_min'], ell2_max=bas_c['ell2_max'],
        n_walkers=mc_c['n_walkers'], step_size=mc_c['step_size'],
        n_burn=mc_c['n_burn'], n_burn_continue=n_burn_continue,
        n_meas=mc_c['n_meas'], thin=mc_c['thin'],
        diag_shift=tdvp_c['diag_shift'], eig_cutoff=tdvp_c['eig_cutoff'],
        omega_floor_rho=tdvp_c['omega_floor'], omega_floor_z=tdvp_c['omega_floor'],
        max_param_rms=tdvp_c['max_param_rms'], s_eps=tdvp_c['s_eps']
    )
    
    print(f"\nConfiguration:")
    print(f"  N = {cfg.N}, n_walkers = {cfg.n_walkers}")
    print(f"  nvar = {cfg.nvar} (2 + {cfg.n_rho} + {cfg.n_z} + 2×{cfg.n2})")
    print(f"  n_burn = {cfg.n_burn}, n_burn_continue = {cfg.n_burn_continue}")
    print(f"  Memory optimization: O(N) space via row-wise vmap")
    
    state = init_state(cfg, jrand.PRNGKey(42))
    
    # =========================================================================
    # Warm-up JIT
    # =========================================================================
    print(f"\n{'='*70}")
    print("Warming up JIT compilation...")
    print(f"{'='*70}")
    
    t_warmup = time.perf_counter()
    state, E, _, _, _ = tdvp_step_rk2(state, cfg, sys_c['omega_rho'], sys_c['omega_z'],
                                       1e-4, "imag", first_step=True)
    E.block_until_ready()
    print(f"JIT warm-up took {time.perf_counter() - t_warmup:.1f}s")
    
    # =========================================================================
    # Imaginary-time evolution
    # =========================================================================
    imag_c = evo_c['imaginary_time']
    n_imag_steps = imag_c['steps']
    
    print(f"\n{'='*70}")
    print(f"Imaginary-time evolution ({n_imag_steps} steps)")
    print(f"{'='*70}")
    
    t_start = time.perf_counter()
    tau_accumulated = 0.0
    
    for it in range(n_imag_steps):
        first_step = (it == 0)
        state, E, rho2, z2, dt_actual = tdvp_step_rk4(
            state, cfg, sys_c['omega_rho'], sys_c['omega_z'],
            imag_c['d_tau'], "imag", first_step=first_step
        )
        tau_accumulated += dt_actual
        
        if it % 50 == 0 or it == n_imag_steps - 1:
            omega_eff_rho = cfg.omega_ref_rho + float(state.p[0].real)
            omega_eff_z = cfg.omega_ref_z + float(state.p[1].real)
            aspect = compute_aspect_ratio(float(rho2), float(z2))
            print(f"  it={it:04d}  E={float(E.real):12.6f}  aspect={aspect:.3f}  "
                  f"tau={tau_accumulated:.4f}  ω_ρ={omega_eff_rho:.3f}  ω_z={omega_eff_z:.3f}")
    
    state.X.block_until_ready()
    t_imag = time.perf_counter() - t_start
    
    E_gs = float(E.real)
    rho2_gs = float(rho2)
    z2_gs = float(z2)
    aspect_gs = compute_aspect_ratio(rho2_gs, z2_gs)
    params_gs = np.array(state.p)
    
    print(f"\nGround state: E_gs = {E_gs:.6f}, aspect = {aspect_gs:.3f}")
    print(f"Imaginary-time took {t_imag:.1f}s ({t_imag/n_imag_steps*1000:.2f} ms/step)")
    print(f"Total tau evolved: {tau_accumulated:.4f} (requested: {n_imag_steps * imag_c['d_tau']:.4f})")
    
    # =========================================================================
    # Real-time evolution
    # =========================================================================
    real_c = evo_c['real_time']
    dt = real_c['dt']
    t_max = real_c['t_max']
    n_real_steps = int(t_max / dt)
    snapshot_interval = real_c.get('snapshot_interval', n_real_steps + 1)
    
    print(f"\n{'='*70}")
    print(f"Real-time evolution ({n_real_steps} steps, target t_max={t_max})")
    print(f"Quench: omega_rho {sys_c['omega_rho']} -> {real_c['quench_omega_rho']}, "
          f"omega_z {sys_c['omega_z']} -> {real_c['quench_omega_z']}")
    print(f"{'='*70}")
    
    current_time = 0.0
    times = [current_time]
    energies = [E_gs]
    rho2_vals = [rho2_gs]
    z2_vals = [z2_gs]
    dt_history = []
    
    param_snapshots = [np.array(state.p)]
    snapshot_times = [current_time]
    
    t_start = time.perf_counter()
    print_every = max(1, n_real_steps // 20)
    
    for step in range(n_real_steps):
        first_step = (step == 0)
        state, E, rho2, z2, dt_actual = tdvp_step_rk4(
            state, cfg, real_c['quench_omega_rho'], real_c['quench_omega_z'],
            dt, "real", first_step=first_step
        )
        
        current_time += dt_actual
        dt_history.append(dt_actual)
        
        times.append(current_time)
        energies.append(float(E.real))
        rho2_vals.append(float(rho2))
        z2_vals.append(float(z2))
        
        if (step + 1) % snapshot_interval == 0:
            print(f"  Snapshot at t={current_time:.4f}")
            param_snapshots.append(np.array(state.p))
            snapshot_times.append(current_time)
        
        if (step + 1) % print_every == 0:
            t_nominal = (step + 1) * dt
            dt_ratio = dt_actual / dt if dt > 0 else 1.0
            aspect = compute_aspect_ratio(float(rho2), float(z2))
            print(f"  step={step+1:5d}  t={current_time:.4f} (nom={t_nominal:.4f})  "
                  f"E={float(E.real):12.6f}  aspect={aspect:.3f}  dt_ratio={dt_ratio:.3f}")
    
    state.X.block_until_ready()
    t_real = time.perf_counter() - t_start
    
    # Compute statistics
    dt_history = np.array(dt_history)
    dt_mean = np.mean(dt_history)
    dt_min = np.min(dt_history)
    dt_max_actual = np.max(dt_history)
    n_reduced = np.sum(dt_history < dt * 0.99)
    
    energies_arr = np.array(energies)
    E_mean = np.mean(energies_arr)
    E_std = np.std(energies_arr)
    E_drift = energies_arr[-1] - energies_arr[0]
    
    print(f"\n{'='*70}")
    print("Simulation complete!")
    print(f"{'='*70}")
    print(f"Real-time took {t_real:.1f}s ({t_real/n_real_steps*1000:.2f} ms/step)")
    print(f"\nTime evolution:")
    print(f"  Requested t_max: {t_max:.4f}")
    print(f"  Actual final time: {current_time:.4f} ({100*current_time/t_max:.1f}%)")
    print(f"  Steps with reduced dt: {n_reduced}/{n_real_steps} ({100*n_reduced/n_real_steps:.1f}%)")
    print(f"  dt statistics: mean={dt_mean:.2e}, min={dt_min:.2e}, max={dt_max_actual:.2e}")
    print(f"\nEnergy statistics:")
    print(f"  E_mean = {E_mean:.6f}")
    print(f"  E_std  = {E_std:.6f} ({100*E_std/abs(E_mean):.3f}%)")
    print(f"  E_drift = {E_drift:+.6f} ({100*E_drift/abs(E_gs):.3f}%)")
    
    # Save results
    np.savez(results_path,
             times=np.array(times), energies=energies_arr,
             rho2_values=np.array(rho2_vals), z2_values=np.array(z2_vals),
             dt_history=dt_history,
             E_gs=E_gs, rho2_gs=rho2_gs, z2_gs=z2_gs, aspect_gs=aspect_gs,
             final_params=np.array(state.p),
             params_gs=params_gs,
             param_snapshots=np.array(param_snapshots),
             snapshot_times=np.array(snapshot_times),
             n_rho=cfg.n_rho, n_z=cfg.n_z, n2=cfg.n2,
             ell2_min=cfg.ell2_min, ell2_max=cfg.ell2_max,
             kappa_cusp=cfg.kappa_cusp, eps_cusp=cfg.eps_cusp, rc_cusp=cfg.rc_cusp,
             N=cfg.N, C3=cfg.C3, C6=cfg.C6, w1_w2=cfg.w1_w2, w0_w2=cfg.w0_w2,
             omega_rho=sys_c['omega_rho'], omega_z=sys_c['omega_z'],
             omega_rho_quench=real_c['quench_omega_rho'],
             omega_z_quench=real_c['quench_omega_z'],
             t_max_requested=t_max,
             t_max_actual=current_time)
    
    np.save(walkers_path, np.array(state.X))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()