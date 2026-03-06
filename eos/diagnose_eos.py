#!/usr/bin/env python3
"""
Diagnostic tests for uniform_vmc_eos.py

Test 1: Non-interacting limit (C0=C2=0 → Ψ=const → E_kin must be 0)
Test 2: Virial check (T_local vs T_hermitian = 1/2 <|∇U|²>)
Test 3: Finite-difference gradient & laplacian verification
"""

import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random as jrand
import numpy as np

Float = jnp.float32

# Import all functions from uniform_vmc_eos
exec(open('uniform_vmc_eos.py').read())

# Suppress the main() call that happens during exec
# (it only runs if __name__ == '__main__', which it isn't here)


# ==============================================================================
# Test 1: Non-interacting limit
# ==============================================================================
def test_noninteracting():
    """With C0=C2=0 (no Jastrow), Ψ=const, so E_kin must be exactly 0."""
    print("=" * 60)
    print("TEST 1: Non-interacting limit (C0=C2=0)")
    print("=" * 60)

    # Save originals
    C0_orig = jnp.array(C0)
    C2_orig = jnp.array(C2)

    # Monkey-patch to zero
    import uniform_vmc_eos as mod  # won't work with exec, so we do it inline

    # Instead, define a modified energy function with zero Jastrow
    C0_zero = jnp.zeros_like(C0)
    C2_zero = jnp.zeros_like(C2)

    @jit
    def compute_energy_zero_jastrow(pos, L):
        """Energy with C0=C2=0 (only cusp remains)."""
        def particle_derivs(r_i, all_pos):
            diff = minimum_image(r_i - all_pos, L)
            r2 = jnp.sum(diff * diff, axis=-1)
            r = jnp.sqrt(r2 + S_EPS * S_EPS)
            inv_r = 1.0 / r
            dz = diff[:, 2]
            cos_theta = jnp.clip(dz * inv_r, -1.0, 1.0)
            p2 = P2(cos_theta)

            f0, f0p, f0pp, f2, f2p, f2pp = u2_f0_f2_derivs_vec(
                r, C0_zero, C2_zero, ELL2, KAPPA_CUSP, EPS_CUSP, RC_CUSP)

            s2 = r2 + S_EPS * S_EPS
            g_r = (f0p + f2p * p2) * inv_r - 3.0 * f2 * (dz ** 2) / (s2 * s2)
            g_zadd = 3.0 * f2 * dz / s2
            grad_x = diff[:, 0] * g_r
            grad_y = diff[:, 1] * g_r
            grad_z = diff[:, 2] * g_r + g_zadd
            lap_val = (f0pp + 2.0 * f0p * inv_r
                       + (f2pp + 2.0 * f2p * inv_r - 6.0 * f2 / s2) * p2)
            v_mw = V_mw_shielding_vec(
                diff[:, 0], diff[:, 1], diff[:, 2],
                C3, C6, W1_W2, W0_W2, EPS_R)
            mask = r > 2.0 * S_EPS
            total_grad = jnp.stack([
                jnp.sum(jnp.where(mask, grad_x, 0.0)),
                jnp.sum(jnp.where(mask, grad_y, 0.0)),
                jnp.sum(jnp.where(mask, grad_z, 0.0))
            ])
            total_lap = jnp.sum(jnp.where(mask, lap_val, 0.0))
            total_v = jnp.sum(jnp.where(mask, v_mw, 0.0))
            return total_grad, total_lap, total_v

        grad_u2, lap_u2, V_int_pp = vmap(
            particle_derivs, in_axes=(0, None))(pos, pos)
        grad_sq = jnp.sum(grad_u2 * grad_u2, axis=1)
        E_kin = -0.5 * jnp.sum(grad_sq + lap_u2)
        E_int = 0.5 * jnp.sum(V_int_pp)
        return E_kin, E_int, jnp.sum(grad_sq), jnp.sum(lap_u2)

    N = 8
    rho = 1.0
    L = float((N / rho) ** (1.0 / 3.0))

    key = jrand.PRNGKey(123)
    X, key = init_walkers_lattice(64, N, L, key)

    # Compute for a few walkers
    for w in range(3):
        E_kin, E_int, grad_sq, lap = compute_energy_zero_jastrow(X[w], L)
        print(f"  Walker {w}: E_kin={float(E_kin):.6e}, "
              f"|∇U|²={float(grad_sq):.6e}, ∇²U={float(lap):.6e}")

    # Note: even with C0=C2=0, the cusp function is still active!
    # cusp_u has kappa=0.1, eps=0.005, rc=0.02
    # For well-separated particles (r >> rc=0.02), cusp ≈ 0
    # So E_kin should be ~0 for lattice-initialized particles at moderate density
    print("  (Cusp still active but negligible for well-separated particles)")
    print()


# ==============================================================================
# Test 2: Virial identity check
# ==============================================================================
def test_virial():
    """
    Check the identity <T> = 1/2 <|∇U|²> = -1/2 <|∇U|² + ∇²U>.

    For each walker, compute:
      T_local = -1/2 * sum_i (|∇_i U|² + ∇²_i U)
      T_hermitian = 1/2 * sum_i |∇_i U|²

    These should be equal on average (and T_hermitian is always ≥ 0).
    """
    print("=" * 60)
    print("TEST 2: Virial identity T_local vs T_hermitian")
    print("=" * 60)

    @jit
    def compute_energy_decomposed(pos, L):
        """Return grad_sq_total, lap_total, E_int separately."""
        def particle_derivs(r_i, all_pos):
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
            lap_val = (f0pp + 2.0 * f0p * inv_r
                       + (f2pp + 2.0 * f2p * inv_r - 6.0 * f2 / s2) * p2)
            v_mw = V_mw_shielding_vec(
                diff[:, 0], diff[:, 1], diff[:, 2],
                C3, C6, W1_W2, W0_W2, EPS_R)
            mask = r > 2.0 * S_EPS
            total_grad = jnp.stack([
                jnp.sum(jnp.where(mask, grad_x, 0.0)),
                jnp.sum(jnp.where(mask, grad_y, 0.0)),
                jnp.sum(jnp.where(mask, grad_z, 0.0))
            ])
            total_lap = jnp.sum(jnp.where(mask, lap_val, 0.0))
            total_v = jnp.sum(jnp.where(mask, v_mw, 0.0))
            return total_grad, total_lap, total_v

        grad_u2, lap_u2, V_int_pp = vmap(
            particle_derivs, in_axes=(0, None))(pos, pos)
        grad_sq_total = jnp.sum(grad_u2 * grad_u2)  # Σ_i |∇_i U|²
        lap_total = jnp.sum(lap_u2)  # Σ_i ∇²_i U
        E_int = 0.5 * jnp.sum(V_int_pp)
        return grad_sq_total, lap_total, E_int

    N = 8
    key = jrand.PRNGKey(42)

    for rho in [0.1, 0.5, 1.0, 2.0, 3.0]:
        L = float((N / rho) ** (1.0 / 3.0))
        n_walkers = 256

        X, key = init_walkers_lattice(n_walkers, N, L, key)

        # Equilibrate briefly
        step = 0.3 * L
        for _ in range(20):
            key, k1, k2 = jrand.split(key, 3)
            disp = jrand.uniform(k1, (n_walkers, N, 3), dtype=Float,
                                 minval=-0.5*step, maxval=0.5*step)
            u = jrand.uniform(k2, (n_walkers, N), dtype=Float)
            X, _ = metropolis_sweep_batch(X, N, L, disp, u)

        # Compute decomposed energies for all walkers
        grad_sq_all, lap_all, E_int_all = vmap(
            compute_energy_decomposed, in_axes=(0, None))(X, L)

        mean_grad_sq = float(jnp.mean(grad_sq_all))
        mean_lap = float(jnp.mean(lap_all))

        T_hermitian = 0.5 * mean_grad_sq  # Always ≥ 0
        T_local = -0.5 * (mean_grad_sq + mean_lap)

        print(f"  rho={rho:.1f}, L={L:.3f}:")
        print(f"    <|∇U|²>  = {mean_grad_sq:+.6f}")
        print(f"    <∇²U>    = {mean_lap:+.6f}")
        print(f"    T_hermit = {T_hermitian/N:+.6f} /N  (must be ≥ 0)")
        print(f"    T_local  = {T_local/N:+.6f} /N  (should equal T_hermit)")
        print(f"    Ratio T_local/T_hermit = {T_local/T_hermitian:.4f}  "
              f"(should be 1.0)")
        print()


# ==============================================================================
# Test 3: Finite-difference gradient & laplacian check
# ==============================================================================
def test_finite_differences():
    """
    Compare analytical gradient and laplacian with finite differences
    of the log-psi function.
    """
    print("=" * 60)
    print("TEST 3: Finite-difference gradient & laplacian check")
    print("=" * 60)

    @jit
    def compute_log_psi(pos, L):
        """log|Ψ| = Σ_{i<j} u2(r_ij, cosθ_ij) with minimum image."""
        def particle_interaction(r_i, all_pos):
            diff = minimum_image(r_i - all_pos, L)
            r2 = jnp.sum(diff * diff, axis=-1)
            r = jnp.sqrt(r2 + S_EPS * S_EPS)
            inv_r = 1.0 / r
            cos_theta = jnp.clip(diff[:, 2] * inv_r, -1.0, 1.0)
            val = u2_val_vec(r, cos_theta, C0, C2, ELL2,
                             KAPPA_CUSP, EPS_CUSP, RC_CUSP)
            mask = r > 2.0 * S_EPS
            return jnp.sum(jnp.where(mask, val, 0.0))

        u2_per_particle = vmap(particle_interaction, in_axes=(0, None))(pos, pos)
        return 0.5 * jnp.sum(u2_per_particle)

    @jit
    def compute_analytical_derivs(pos, L):
        """Analytical gradient and laplacian from the code."""
        def particle_derivs(r_i, all_pos):
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
            lap_val = (f0pp + 2.0 * f0p * inv_r
                       + (f2pp + 2.0 * f2p * inv_r - 6.0 * f2 / s2) * p2)
            mask = r > 2.0 * S_EPS
            total_grad = jnp.stack([
                jnp.sum(jnp.where(mask, grad_x, 0.0)),
                jnp.sum(jnp.where(mask, grad_y, 0.0)),
                jnp.sum(jnp.where(mask, grad_z, 0.0))
            ])
            total_lap = jnp.sum(jnp.where(mask, lap_val, 0.0))
            return total_grad, total_lap

        grad_u2, lap_u2 = vmap(particle_derivs, in_axes=(0, None))(pos, pos)
        return grad_u2, lap_u2  # (N, 3) and (N,)

    N = 8
    eps_fd = 1e-4  # Finite-difference step

    for rho in [0.5, 2.0]:
        L = float((N / rho) ** (1.0 / 3.0))
        key = jrand.PRNGKey(99)
        X, key = init_walkers_lattice(1, N, L, key)
        pos = X[0]  # Single configuration (N, 3)

        # Analytical
        grad_anal, lap_anal = compute_analytical_derivs(pos, L)
        grad_anal = np.array(grad_anal)  # (N, 3)
        lap_anal = np.array(lap_anal)    # (N,)

        # Numerical gradient and laplacian
        grad_num = np.zeros((N, 3))
        lap_num = np.zeros(N)

        log_psi_0 = float(compute_log_psi(pos, L))

        for i in range(N):
            for alpha in range(3):
                pos_plus = np.array(pos)
                pos_minus = np.array(pos)
                pos_plus[i, alpha] += eps_fd
                pos_minus[i, alpha] -= eps_fd

                lp_plus = float(compute_log_psi(jnp.array(pos_plus, dtype=Float), L))
                lp_minus = float(compute_log_psi(jnp.array(pos_minus, dtype=Float), L))

                grad_num[i, alpha] = (lp_plus - lp_minus) / (2 * eps_fd)
                lap_num[i] += (lp_plus - 2 * log_psi_0 + lp_minus) / (eps_fd ** 2)

        print(f"  rho={rho:.1f}, L={L:.3f}, eps_fd={eps_fd:.0e}:")
        # Compare gradient
        grad_err = np.abs(grad_anal - grad_num)
        grad_rel = grad_err / (np.abs(grad_num) + 1e-20)
        print(f"    Gradient max|error|   = {grad_err.max():.2e}")
        print(f"    Gradient max|rel err| = {grad_rel.max():.2e}")

        # Compare laplacian
        lap_err = np.abs(lap_anal - lap_num)
        lap_rel = lap_err / (np.abs(lap_num) + 1e-20)
        print(f"    Laplacian max|error|   = {lap_err.max():.2e}")
        print(f"    Laplacian max|rel err| = {lap_rel.max():.2e}")

        # Show per-particle details
        for i in range(min(3, N)):
            print(f"    Particle {i}:")
            print(f"      grad anal = [{grad_anal[i,0]:+.6f}, "
                  f"{grad_anal[i,1]:+.6f}, {grad_anal[i,2]:+.6f}]")
            print(f"      grad num  = [{grad_num[i,0]:+.6f}, "
                  f"{grad_num[i,1]:+.6f}, {grad_num[i,2]:+.6f}]")
            print(f"      lap anal  = {lap_anal[i]:+.6f}")
            print(f"      lap num   = {lap_num[i]:+.6f}")

        # Compute T_local and T_hermitian for this single config
        grad_sq = np.sum(grad_anal ** 2)
        T_local = -0.5 * (grad_sq + np.sum(lap_anal))
        T_hermit = 0.5 * grad_sq
        print(f"    T_local/N    = {T_local/N:+.6f}")
        print(f"    T_hermitian/N = {T_hermit/N:+.6f}")
        print()


# ==============================================================================
# Test 4: Check log-psi continuity across minimum-image boundary
# ==============================================================================
def test_minimum_image_continuity():
    """
    Move a particle across the minimum-image boundary and check
    if log_psi and its derivatives are continuous.
    """
    print("=" * 60)
    print("TEST 4: Minimum-image boundary continuity check")
    print("=" * 60)

    @jit
    def compute_log_psi(pos, L):
        def particle_interaction(r_i, all_pos):
            diff = minimum_image(r_i - all_pos, L)
            r2 = jnp.sum(diff * diff, axis=-1)
            r = jnp.sqrt(r2 + S_EPS * S_EPS)
            inv_r = 1.0 / r
            cos_theta = jnp.clip(diff[:, 2] * inv_r, -1.0, 1.0)
            val = u2_val_vec(r, cos_theta, C0, C2, ELL2,
                             KAPPA_CUSP, EPS_CUSP, RC_CUSP)
            mask = r > 2.0 * S_EPS
            return jnp.sum(jnp.where(mask, val, 0.0))
        u2_per_particle = vmap(particle_interaction, in_axes=(0, None))(pos, pos)
        return 0.5 * jnp.sum(u2_per_particle)

    N = 8
    rho = 2.0
    L = float((N / rho) ** (1.0 / 3.0))

    key = jrand.PRNGKey(7)
    X, key = init_walkers_lattice(1, N, L, key)
    pos = np.array(X[0])

    # Place particle 0 at various x-positions, scanning across the
    # minimum-image boundary relative to particle 1
    particle_1_x = pos[1, 0]
    boundary_x = particle_1_x + L / 2  # This is where MI boundary is

    x_scan = np.linspace(boundary_x - 0.2, boundary_x + 0.2, 41)
    log_psis = []
    for x in x_scan:
        pos_test = pos.copy()
        pos_test[0, 0] = x
        lp = float(compute_log_psi(jnp.array(pos_test, dtype=Float), L))
        log_psis.append(lp)

    log_psis = np.array(log_psis)
    d_log_psi = np.diff(log_psis) / np.diff(x_scan)

    print(f"  Scanning particle 0 x-position across MI boundary")
    print(f"  rho={rho:.1f}, L={L:.3f}, boundary at x={boundary_x:.3f}")
    print(f"  log_psi range: [{log_psis.min():.6f}, {log_psis.max():.6f}]")
    print(f"  d(log_psi)/dx range: [{d_log_psi.min():.6f}, {d_log_psi.max():.6f}]")

    # Check for discontinuity in derivative
    mid = len(d_log_psi) // 2
    grad_before = d_log_psi[mid - 2:mid]
    grad_after = d_log_psi[mid:mid + 2]
    print(f"  Gradient just before boundary: {grad_before}")
    print(f"  Gradient just after boundary:  {grad_after}")
    jump = np.abs(grad_after[0] - grad_before[-1])
    print(f"  Gradient jump at boundary: {jump:.6f}")
    if jump > 0.01:
        print("  WARNING: Significant gradient discontinuity at MI boundary!")
        print("  This can cause incorrect local kinetic energy.")
    print()


if __name__ == '__main__':
    test_noninteracting()
    test_virial()
    test_finite_differences()
    test_minimum_image_continuity()
