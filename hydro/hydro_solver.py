#!/usr/bin/env python3
"""
Phase 4: Superfluid hydrodynamic PDE solver for trap-quench dynamics.

Solves the Gross-Pitaevskii equation for ψ = √ρ e^{iθ} in cylindrical
coordinates (r, z) with azimuthal symmetry:

    i∂ψ/∂t = [-½∇² + V(r,z,t) + μ(|ψ|²)] ψ

Uses the EOS from Phase 3 VMC fits:
  ε(ρ) = a1 ρ + a2 ρ² + a3 ρ³
  μ(ρ) = 2a1 ρ + 3a2 ρ² + 4a3 ρ³

Initial conditions:
  - 'gaussian': matched to tVMC moments ⟨ρ²⟩, ⟨z²⟩
  - 'gs': GPE ground state via imaginary-time propagation
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────────────────────────────────
# EOS with high-density regularization
# ──────────────────────────────────────────────────────────────────────────────

def load_eos(eos_file):
    d = np.load(eos_file, allow_pickle=True)
    return d['poly3_coeffs']


class EOS:
    """EOS ε(ρ)=a1 ρ+a2 ρ²+a3 ρ³, linearly extended above ρ_reg."""

    def __init__(self, coeffs, reg_frac=0.80, min_slope=None):
        self.a1, self.a2, self.a3 = coeffs
        self.coeffs = coeffs
        roots = np.roots([12*self.a3, 6*self.a2, 2*self.a1])
        pos = [r.real for r in roots if r.real > 0 and abs(r.imag) < 1e-10]
        self.rho_stab = max(pos) if pos else np.inf
        self.rho_reg = self.rho_stab * reg_frac
        rr = self.rho_reg
        self.mu_at_reg = 2*self.a1*rr + 3*self.a2*rr**2 + 4*self.a3*rr**3
        nat_slope = 2*self.a1 + 6*self.a2*rr + 12*self.a3*rr**2
        if min_slope is None:
            min_slope = 2*self.a1
        self.dmu_at_reg = max(nat_slope, min_slope)
        self.eps_at_reg = self.a1*rr + self.a2*rr**2 + self.a3*rr**3

    def eps(self, rho):
        out = np.empty_like(rho)
        lo = rho <= self.rho_reg
        r_lo = rho[lo]
        out[lo] = self.a1*r_lo + self.a2*r_lo**2 + self.a3*r_lo**3
        r_hi = rho[~lo]
        dr = r_hi - self.rho_reg
        rho_eps = (self.rho_reg*self.eps_at_reg
                   + self.mu_at_reg*dr + 0.5*self.dmu_at_reg*dr**2)
        out[~lo] = rho_eps / r_hi
        return out

    def mu(self, rho):
        out = np.empty_like(rho)
        lo = rho <= self.rho_reg
        r_lo = rho[lo]
        out[lo] = 2*self.a1*r_lo + 3*self.a2*r_lo**2 + 4*self.a3*r_lo**3
        out[~lo] = self.mu_at_reg + self.dmu_at_reg*(rho[~lo] - self.rho_reg)
        return out

    def dmu(self, rho):
        out = np.empty_like(rho)
        lo = rho <= self.rho_reg
        r_lo = rho[lo]
        out[lo] = 2*self.a1 + 6*self.a2*r_lo + 12*self.a3*r_lo**2
        out[~lo] = self.dmu_at_reg
        return np.maximum(out, 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Grid and integration
# ──────────────────────────────────────────────────────────────────────────────

def make_grid(N_r, N_z, R_max, Z_max):
    r = np.linspace(0, R_max, N_r)
    z = np.linspace(-Z_max, Z_max, N_z)
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    R, Z = np.meshgrid(r, z, indexing='ij')
    return r, z, dr, dz, R, Z


def integrate_cyl(f, r, z, dr, dz):
    """∫ f(r,z) 2πr dr dz (trapezoidal)."""
    wr = 2*np.pi*r*dr
    wr[0] *= 0.5; wr[-1] *= 0.5
    wz = np.full(len(z), dz)
    wz[0] *= 0.5; wz[-1] *= 0.5
    return np.einsum('ij,i,j->', f.real, wr, wz)


def moment_r2(rho, R, r, z, dr, dz, N):
    return integrate_cyl(R**2*rho, r, z, dr, dz) / N


def moment_z2(rho, Z, r, z, dr, dz, N):
    return integrate_cyl(Z**2*rho, r, z, dr, dz) / N


# ──────────────────────────────────────────────────────────────────────────────
# Spatial derivatives (cylindrical)
# ──────────────────────────────────────────────────────────────────────────────

def laplacian_cyl(f, r_1d, dr, dz):
    """∇²f = ∂²f/∂r² + (1/r)∂f/∂r + ∂²f/∂z² in cylindrical coords.

    Uses 4th-order central differences in the interior, 2nd-order at
    boundaries. Boundary conditions:
      r=0: symmetry (ghost f(-r) = f(r))
      r=R_max: Dirichlet f=0
      z=±Z_max: Dirichlet f=0
    """
    Nr, Nz = f.shape
    dr2 = dr**2
    dz2 = dz**2

    # ── d²f/dr² (4th order interior, symmetry at r=0, Dirichlet at R_max)
    d2r = np.zeros_like(f)
    # Interior: 4th-order stencil (-f[i-2]+16f[i-1]-30f[i]+16f[i+1]-f[i+2])/(12*dr²)
    d2r[2:-2] = (-f[4:] + 16*f[3:-1] - 30*f[2:-2]
                 + 16*f[1:-3] - f[:-4]) / (12*dr2)
    # i=0: ghost f(-dr)=f(1), f(-2dr)=f(2)
    d2r[0] = (-2*f[2] + 32*f[1] - 30*f[0]) / (12*dr2)
    # i=1: ghost f(-dr)=f(1) → stencil uses f(-dr)=f(1)
    d2r[1] = (-f[3] + 16*f[2] - 30*f[1] + 16*f[0] - f[1]) / (12*dr2)
    # i=N-2: f(N)=0 (Dirichlet ghost)
    d2r[-2] = (16*f[-1] - 30*f[-2] + 16*f[-3] - f[-4]) / (12*dr2)
    # i=N-1: f(N)=f(N+1)=0 (Dirichlet)
    d2r[-1] = (-30*f[-1] + 16*f[-2] - f[-3]) / (12*dr2)

    # ── (1/r) df/dr (4th order interior)
    dfr = np.zeros_like(f)
    # Interior: 4th-order (-f[i+2]+8f[i+1]-8f[i-1]+f[i-2])/(12*dr)
    dfr[2:-2] = (-f[4:] + 8*f[3:-1] - 8*f[1:-3] + f[:-4]) / (12*dr)
    # i=0: df/dr=0 by symmetry
    dfr[0] = 0.0
    # i=1: ghost f(-dr)=f(1) → (-f[3]+8f[2]-8f[0]+f[1])/(12*dr)
    dfr[1] = (-f[3] + 8*f[2] - 8*f[0] + f[1]) / (12*dr)
    # i=N-2: f(N)=0 → (-0+8f[N-1]-8f[N-3]+f[N-4])/(12*dr)
    dfr[-2] = (8*f[-1] - 8*f[-3] + f[-4]) / (12*dr)
    # i=N-1: f(N)=f(N+1)=0 → (-8f[-2]+f[-3])/(12*dr)
    dfr[-1] = (-8*f[-2] + f[-3]) / (12*dr)

    one_r_dfr = np.zeros_like(f)
    one_r_dfr[1:] = dfr[1:] / r_1d[1:, np.newaxis]
    one_r_dfr[0] = d2r[0]  # L'Hôpital: (1/r)df/dr|_0 = d²f/dr²|_0

    # ── d²f/dz² (4th order interior, Dirichlet at ±Z_max)
    d2z = np.zeros_like(f)
    d2z[:, 2:-2] = (-f[:, 4:] + 16*f[:, 3:-1] - 30*f[:, 2:-2]
                    + 16*f[:, 1:-3] - f[:, :-4]) / (12*dz2)
    # j=0: f(z_min-dz)=0, f(z_min-2dz)=0 (Dirichlet)
    d2z[:, 0] = (-f[:, 2] + 16*f[:, 1] - 30*f[:, 0]) / (12*dz2)
    # j=1: f(z_min-dz)=0
    d2z[:, 1] = (-f[:, 3] + 16*f[:, 2] - 30*f[:, 1] + 16*f[:, 0]) / (12*dz2)
    # j=N-2: f(z_max+dz)=0
    d2z[:, -2] = (16*f[:, -1] - 30*f[:, -2] + 16*f[:, -3]
                  - f[:, -4]) / (12*dz2)
    # j=N-1: f(z_max+dz)=0, f(z_max+2dz)=0
    d2z[:, -1] = (-f[:, -3] + 16*f[:, -2] - 30*f[:, -1]) / (12*dz2)

    return d2r + one_r_dfr + d2z


def grad_sq(f, dr, dz):
    """Returns |∇f|² in cylindrical coords (no azimuthal component)."""
    dfr = np.zeros_like(f)
    dfr[1:-1] = (f[2:] - f[:-2]) / (2*dr)
    dfr[0] = 0.0
    dfr[-1] = (f[-1] - f[-2]) / dr
    dfz = np.zeros_like(f)
    dfz[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dz)
    dfz[:, 0] = (f[:, 1] - f[:, 0]) / dz
    dfz[:, -1] = (f[:, -1] - f[:, -2]) / dz
    return dfr**2 + dfz**2


# ──────────────────────────────────────────────────────────────────────────────
# Initial conditions
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_ic(r, z, dr, dz, R, Z, N_part, rho2_target, z2_target):
    """Gaussian ρ matched to tVMC moments. Returns ψ = √ρ (real)."""
    sr2, sz2 = rho2_target / 2.0, z2_target
    rho0 = np.exp(-R**2/(2*sr2) - Z**2/(2*sz2))
    rho0 *= N_part / integrate_cyl(rho0, r, z, dr, dz)
    psi = np.sqrt(rho0).astype(complex)
    print(f"  Gaussian IC: σ_r²={sr2:.4f}, σ_z²={sz2:.4f}")
    print(f"  ρ_peak={rho0.max():.4f}, N={integrate_cyl(rho0, r, z, dr, dz):.6f}")
    print(f"  ⟨ρ²⟩={moment_r2(rho0, R, r, z, dr, dz, N_part):.4f}, "
          f"⟨z²⟩={moment_z2(rho0, Z, r, z, dr, dz, N_part):.4f}")
    return psi


def groundstate_ic(r, z, dr, dz, R, Z, N_part, V, eos,
                   dtau=None, tol=1e-8, max_iter=50000):
    """GPE ground state via imaginary-time propagation.

    Propagates ψ → ψ − dτ·Hψ and renormalizes to N_part each step.
    Converges when the chemical potential μ = ⟨H⟩/N stabilizes.
    """
    print("  Computing GPE ground state (imaginary-time propagation)...")

    dx = min(dr, dz)
    if dtau is None:
        dtau = 0.3 * dx**2 / np.pi  # dispersive CFL for stability

    # Initial guess: Gaussian
    sr2, sz2 = 0.5, 0.5
    rho0 = np.exp(-R**2/(2*sr2) - Z**2/(2*sz2))
    rho0 *= N_part / integrate_cyl(rho0, r, z, dr, dz)
    psi = np.sqrt(rho0).astype(complex)

    mu_prev = 1e10
    for it in range(max_iter):
        # H·ψ = -½∇²ψ + V·ψ + μ(|ψ|²)·ψ
        rho = np.abs(psi)**2
        Hpsi = -0.5 * laplacian_cyl(psi, r, dr, dz) + V * psi + eos.mu(rho) * psi

        # Imaginary-time step: ψ → ψ − dτ·Hψ
        psi = psi - dtau * Hpsi

        # Renormalize to N_part
        rho = np.abs(psi)**2
        Nt = integrate_cyl(rho, r, z, dr, dz)
        if Nt <= 0:
            print(f"  WARNING: N collapsed at iteration {it}")
            break
        psi *= np.sqrt(N_part / Nt)

        # Check convergence every 100 steps
        if it % 100 == 0:
            rho = np.abs(psi)**2
            Hpsi = (-0.5 * laplacian_cyl(psi, r, dr, dz)
                    + V * psi + eos.mu(rho) * psi)
            mu_cur = integrate_cyl((psi.conj() * Hpsi).real,
                                   r, z, dr, dz) / N_part
            dmu = abs(mu_cur - mu_prev)
            if it % 2000 == 0:
                r2 = moment_r2(rho, R, r, z, dr, dz, N_part)
                z2 = moment_z2(rho, Z, r, z, dr, dz, N_part)
                print(f"    iter={it:6d}  μ={mu_cur:.6f}  δμ={dmu:.2e}  "
                      f"ρ_max={rho.max():.4f}  ⟨ρ²⟩={r2:.4f}  ⟨z²⟩={z2:.4f}")
            if dmu < tol and it > 200:
                print(f"  Converged at iter {it}: μ={mu_cur:.6f}, δμ={dmu:.2e}")
                break
            mu_prev = mu_cur

    rho = np.abs(psi)**2
    N_final = integrate_cyl(rho, r, z, dr, dz)
    r2 = moment_r2(rho, R, r, z, dr, dz, N_part)
    z2 = moment_z2(rho, Z, r, z, dr, dz, N_part)
    # Compute energy
    Hpsi = (-0.5 * laplacian_cyl(psi, r, dr, dz) + V * psi
            + eos.mu(rho) * psi)
    mu_final = integrate_cyl((psi.conj() * Hpsi).real, r, z, dr, dz) / N_part
    lap_psi = laplacian_cyl(psi, r, dr, dz)
    E_kin = -0.5 * integrate_cyl((psi.conj() * lap_psi).real, r, z, dr, dz)
    E_pot = integrate_cyl(rho * V, r, z, dr, dz)
    E_int = integrate_cyl(rho * eos.eps(rho), r, z, dr, dz)

    print(f"  Ground state: N={N_final:.6f}, μ={mu_final:.6f}")
    print(f"  ρ_peak={rho.max():.4f}")
    print(f"  ⟨ρ²⟩={r2:.4f}, ⟨z²⟩={z2:.4f}")
    print(f"  E_kin={E_kin:.4f}, E_pot={E_pot:.4f}, E_int={E_int:.4f}, "
          f"E_tot={E_kin+E_pot+E_int:.4f}")
    return psi


# ──────────────────────────────────────────────────────────────────────────────
# GPE solver: i ∂ψ/∂t = [-½∇² + V + μ(|ψ|²)] ψ
# ──────────────────────────────────────────────────────────────────────────────

def gpe_rhs(psi, r_1d, dr, dz, V, eos, include_kinetic=True):
    """Returns ∂ψ/∂t = -i·H·ψ for the GPE."""
    rho = np.abs(psi)**2
    mu_val = eos.mu(rho)
    Hpsi = V * psi + mu_val * psi
    if include_kinetic:
        Hpsi = Hpsi - 0.5 * laplacian_cyl(psi, r_1d, dr, dz)
    return -1j * Hpsi


def gpe_energy(psi, r_1d, z_1d, dr, dz, V, eos, include_kinetic=True):
    """E = ∫[½|∇ψ|² + V|ψ|² + |ψ|²ε(|ψ|²)] 2πr dr dz.

    Kinetic energy computed as -½ Re∫ψ*∇²ψ dV using the same discrete
    Laplacian as the dynamics, ensuring consistent conservation.
    """
    rho = np.abs(psi)**2
    E_pot = rho * V
    E_int = rho * eos.eps(rho)
    e = E_pot + E_int
    if include_kinetic:
        lap_psi = laplacian_cyl(psi, r_1d, dr, dz)
        e_kin = -0.5 * (psi.conj() * lap_psi).real
        e = e + e_kin
    return integrate_cyl(e, r_1d, z_1d, dr, dz)


def gpe_rk4(psi, r_1d, dr, dz, V, eos, dt, include_kinetic=True):
    """RK4 step for the GPE."""
    kw = (r_1d, dr, dz, V, eos, include_kinetic)
    k1 = gpe_rhs(psi, *kw)
    k2 = gpe_rhs(psi + 0.5*dt*k1, *kw)
    k3 = gpe_rhs(psi + 0.5*dt*k2, *kw)
    k4 = gpe_rhs(psi + dt*k3, *kw)
    return psi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def gpe_cfl(psi, r_1d, dr, dz, eos, cfl_factor, include_kinetic=True):
    """CFL for GPE: dispersive (kinetic) + advective (potential)."""
    dx = min(dr, dz)
    rho = np.abs(psi)**2
    rho_max = rho.max()
    mu_max = eos.mu(np.array([rho_max]))[0]
    V_eff_max = max(mu_max, 1.0)
    dt_pot = cfl_factor * 2.0 / V_eff_max

    if include_kinetic:
        dt_disp = cfl_factor * dx**2 / np.pi
        return min(dt_pot, dt_disp)
    else:
        return dt_pot


# ──────────────────────────────────────────────────────────────────────────────
# Scaling ansatz ODE
# ──────────────────────────────────────────────────────────────────────────────

def solve_scaling(wr0, wz0, wr1, wz1, rho2_0, z2_0, t_max):
    Cr, Cz = wr0**2, wz0**2
    def rhs(t, y):
        lr, lz, dlr, dlz = y
        return [dlr, dlz,
                -wr1**2*lr + Cr/(lr**3*lz),
                -wz1**2*lz + Cz/(lr**2*lz**2)]
    sol = solve_ivp(rhs, [0, t_max], [1, 1, 0, 0],
                    method='RK45', max_step=0.001, dense_output=True)
    t = np.linspace(0, t_max, 3000)
    y = sol.sol(t)
    return t, rho2_0*y[0]**2, z2_0*y[1]**2


# ──────────────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────────────

def run_hydro(args):
    print("=" * 60)
    print("Phase 4: Superfluid Hydrodynamic PDE Solver")
    print("=" * 60)

    coeffs = load_eos(args.eos_file)
    eos = EOS(coeffs)
    print(f"\nEOS: a1={eos.a1:.6f}, a2={eos.a2:.6f}, a3={eos.a3:.6f}")
    print(f"  ρ_stab={eos.rho_stab:.3f}, ρ_reg={eos.rho_reg:.3f}, "
          f"dμ/dρ_reg={eos.dmu_at_reg:.4f}")

    N_part = 8
    wr0, wz0 = 1.0, 1.0
    wr1, wz1 = 2.0, 1.0

    r, z, dr, dz, R, Z = make_grid(args.N_r, args.N_z, args.R_max, args.Z_max)
    print(f"\nGrid: {args.N_r}×{args.N_z}, dr={dr:.5f}, dz={dz:.5f}")

    V_pre = 0.5*(wr0**2*R**2 + wz0**2*Z**2)
    V_post = 0.5*(wr1**2*R**2 + wz1**2*Z**2)

    include_kin = args.quantum_pressure
    mode = "GPE (with QP)" if include_kin else "GPE (potential only, no QP)"
    print(f"\nMode: {mode}")

    # Initial condition
    print(f"\nIC: {args.ic}")
    if args.ic == 'gs':
        psi = groundstate_ic(r, z, dr, dz, R, Z, N_part, V_pre, eos)
    else:
        psi = gaussian_ic(r, z, dr, dz, R, Z, N_part, 1.05, 0.525)

    rho0 = np.abs(psi)**2
    N0 = integrate_cyl(rho0, r, z, dr, dz)
    r2_0 = moment_r2(rho0, R, r, z, dr, dz, N_part)
    z2_0 = moment_z2(rho0, Z, r, z, dr, dz, N_part)
    E_pre = gpe_energy(psi, r, z, dr, dz, V_pre, eos, include_kin)
    E_post = gpe_energy(psi, r, z, dr, dz, V_post, eos, include_kin)

    print(f"\nE_pre={E_pre:.4f}, E_post={E_post:.4f}, ΔE={E_post-E_pre:.4f}")
    print(f"Expected ΔE = {0.5*(wr1**2-wr0**2)*r2_0*N_part:.4f}")

    # Time loop
    print(f"\nt_max={args.t_max}, CFL={args.cfl}")

    t_cur = 0.0
    step = 0
    ts = [0.0]; r2s = [r2_0]; z2s = [z2_0]; Ns = [N0]; Es = [E_post]
    last_p = -1.0
    V = V_post

    # Density snapshots
    snap_interval = args.t_max / 15
    next_snap = snap_interval
    snapshots = [(0.0, rho0.copy())]

    while t_cur < args.t_max:
        dt = gpe_cfl(psi, r, dr, dz, eos, args.cfl, include_kin)
        dt = min(dt, args.t_max - t_cur, 0.01)

        psi = gpe_rk4(psi, r, dr, dz, V, eos, dt, include_kin)
        t_cur += dt
        step += 1

        if np.any(np.isnan(psi)):
            print(f"  NaN at t={t_cur:.6f}, step={step}")
            break

        if step % 10 == 0 or t_cur >= args.t_max - 1e-10:
            rho = np.abs(psi)**2
            Nt = integrate_cyl(rho, r, z, dr, dz)
            r2t = moment_r2(rho, R, r, z, dr, dz, N_part)
            z2t = moment_z2(rho, Z, r, z, dr, dz, N_part)
            Et = gpe_energy(psi, r, z, dr, dz, V, eos, include_kin)
            ts.append(t_cur); r2s.append(r2t)
            z2s.append(z2t); Ns.append(Nt); Es.append(Et)

        # Density snapshots
        if t_cur >= next_snap:
            rho = np.abs(psi)**2
            snapshots.append((t_cur, rho.copy()))
            next_snap += snap_interval

        if t_cur - last_p >= 0.3:
            rho = np.abs(psi)**2
            Nt = integrate_cyl(rho, r, z, dr, dz)
            r2t = moment_r2(rho, R, r, z, dr, dz, N_part)
            z2t = moment_z2(rho, Z, r, z, dr, dz, N_part)
            Et = gpe_energy(psi, r, z, dr, dz, V, eos, include_kin)
            dN = abs(Nt - N0) / N0
            dE = abs(Et - E_post) / max(abs(E_post), 1e-10)
            print(f"  t={t_cur:.3f}  ⟨ρ²⟩={r2t:.4f}  ⟨z²⟩={z2t:.4f}  "
                  f"ρ_max={rho.max():.3f}  dt={dt:.6f}  "
                  f"δN={dN:.2e}  δE={dE:.2e}  step={step}")
            last_p = t_cur

    # Final snapshot
    rho = np.abs(psi)**2
    snapshots.append((t_cur, rho.copy()))

    print(f"\nDone: {step} steps")

    ts = np.array(ts); r2s = np.array(r2s); z2s = np.array(z2s)
    Ns = np.array(Ns); Es = np.array(Es)

    # Conservation summary
    dN_max = np.max(np.abs(Ns - N0) / N0)
    dE_max = np.max(np.abs(Es - E_post) / max(abs(E_post), 1e-10))
    print(f"\nConservation: max |δN/N| = {dN_max:.2e}, max |δE/E| = {dE_max:.2e}")

    # Scaling ansatz
    print("\nScaling ansatz...")
    t_sc, r2_sc, z2_sc = solve_scaling(wr0, wz0, wr1, wz1, r2_0, z2_0,
                                       args.t_max)

    # Save results
    print(f"Saving to {args.output}")
    snap_times = np.array([s[0] for s in snapshots])
    snap_densities = np.array([s[1] for s in snapshots])
    np.savez(args.output,
             times=ts, rho2=r2s, z2=z2s, energy=Es, particle_number=Ns,
             rho2_0=r2_0, z2_0=z2_0, E_pre=E_pre, E_post=E_post,
             snapshot_times=snap_times, snapshot_densities=snap_densities,
             r_grid=r, z_grid=z,
             scaling_times=t_sc, scaling_rho2=r2_sc, scaling_z2=z2_sc,
             coeffs=coeffs, ic_type=args.ic,
             quantum_pressure=args.quantum_pressure,
             N_r=args.N_r, N_z=args.N_z,
             R_max=args.R_max, Z_max=args.Z_max,
             t_max=args.t_max, cfl=args.cfl)

    if not args.no_plot:
        make_plots(args, ts, r2s, z2s, Es, Ns, t_sc, r2_sc, z2_sc, E_post)

    return ts, r2s, z2s, Es


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def make_plots(args, times, rho2, z2, energy, N_arr,
               t_sc, rho2_sc, z2_sc, E0_post):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tvmc = None
    try:
        tvmc = np.load(args.tvmc_file, allow_pickle=True)
    except FileNotFoundError:
        print(f"Warning: tVMC file not found: {args.tvmc_file}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    ax.plot(times, rho2, 'b-', lw=2, label='Hydro PDE')
    ax.plot(t_sc, rho2_sc, 'g--', lw=1.5, label='Scaling ansatz')
    if tvmc is not None:
        ax.plot(tvmc['times'], tvmc['rho2_values'], 'r-', lw=1, alpha=0.7,
                label='tVMC')
    ax.set_ylabel(r'$\langle \rho^2 \rangle$')
    ax.set_title(r'Breathing mode: $\omega_\rho: 1 \to 2$'
                 + (' (with QP)' if args.quantum_pressure else ' (no QP)'))
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, z2, 'b-', lw=2, label='Hydro PDE')
    ax.plot(t_sc, z2_sc, 'g--', lw=1.5, label='Scaling ansatz')
    if tvmc is not None:
        ax.plot(tvmc['times'], tvmc['z2_values'], 'r-', lw=1, alpha=0.7,
                label='tVMC')
    ax.set_ylabel(r'$\langle z^2 \rangle$')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    dN = np.abs(N_arr - N_arr[0]) / N_arr[0]
    dE = np.abs(energy - E0_post) / max(abs(E0_post), 1e-10)
    ax.semilogy(times, np.maximum(dN, 1e-16), 'b-', lw=1.5,
                label=r'$|\delta N|/N$')
    ax.semilogy(times, np.maximum(dE, 1e-16), 'r-', lw=1.5,
                label=r'$|\delta E|/E$')
    ax.set_ylabel('Relative error')
    ax.set_xlabel('Time')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('Conservation')

    plt.tight_layout()
    plt.savefig('hydro_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to hydro_comparison.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Superfluid hydrodynamic PDE solver for trap quench')
    p.add_argument('--eos_file', default='eos_fit_fixed.npz')
    p.add_argument('--tvmc_file',
                   default='tvmc_output_N8_freeze_u2/results.npz')
    p.add_argument('--t_max', type=float, default=3.0)
    p.add_argument('--N_r', type=int, default=128)
    p.add_argument('--N_z', type=int, default=256)
    p.add_argument('--R_max', type=float, default=5.0)
    p.add_argument('--Z_max', type=float, default=5.0)
    p.add_argument('--ic', choices=['gs', 'gaussian'], default='gaussian')
    p.add_argument('--quantum_pressure', action='store_true', default=True,
                   help='Include kinetic (∇²) term — full GPE (default: on)')
    p.add_argument('--no_quantum_pressure', dest='quantum_pressure',
                   action='store_false',
                   help='Disable quantum pressure (potential-only evolution)')
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--output', default='hydro_results.npz')
    p.add_argument('--no_plot', action='store_true')
    args = p.parse_args()
    run_hydro(args)


if __name__ == '__main__':
    main()
