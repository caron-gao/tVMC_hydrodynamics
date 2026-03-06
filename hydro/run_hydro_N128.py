#!/usr/bin/env python3
"""
Hydrodynamic (GPE) solver for N=128 trap quench, to compare with tVMC.

Uses the EOS from the frozen-u2 VMC (same interaction potential),
extended to rho=15 for the higher densities encountered at N=128.
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
import sys
import time as time_mod

# ──────────────────────────────────────────────────────────────────────────────
# Import core routines from hydro_solver
# ──────────────────────────────────────────────────────────────────────────────
from hydro_solver import (
    EOS, load_eos, make_grid, integrate_cyl, moment_r2, moment_z2,
    laplacian_cyl, grad_sq,
    gpe_rhs, gpe_energy, gpe_rk4, gpe_cfl,
    solve_scaling,
)


def groundstate_ic_N128(r, z, dr, dz, R, Z, N_part, V, eos,
                        dtau=None, tol=1e-8, max_iter=80000):
    """GPE ground state via imaginary-time propagation for N=128.

    Uses a broader initial Gaussian and longer convergence.
    """
    print("  Computing GPE ground state (imaginary-time propagation)...")
    print(f"  N={N_part}, grid={len(r)}x{len(z)}, domain=[0,{r[-1]:.1f}]x[{z[0]:.1f},{z[-1]:.1f}]")

    dx = min(dr, dz)
    if dtau is None:
        dtau = 0.3 * dx**2 / np.pi

    # Initial guess: broader Gaussian for N=128
    # N=128 in isotropic trap: sigma ~ (N)^(1/5) * a_ho for TF
    # But start with something reasonable
    sr2, sz2 = 0.8, 0.8
    rho0 = np.exp(-R**2/(2*sr2) - Z**2/(2*sz2))
    rho0 *= N_part / integrate_cyl(rho0, r, z, dr, dz)
    psi = np.sqrt(rho0).astype(complex)

    mu_prev = 1e10
    t_start = time_mod.time()
    for it in range(max_iter):
        rho = np.abs(psi)**2
        Hpsi = -0.5 * laplacian_cyl(psi, r, dr, dz) + V * psi + eos.mu(rho) * psi

        psi = psi - dtau * Hpsi

        rho = np.abs(psi)**2
        Nt = integrate_cyl(rho, r, z, dr, dz)
        if Nt <= 0 or np.any(np.isnan(psi)):
            print(f"  WARNING: Collapsed at iteration {it}")
            break
        psi *= np.sqrt(N_part / Nt)

        if it % 100 == 0:
            rho = np.abs(psi)**2
            Hpsi_check = (-0.5 * laplacian_cyl(psi, r, dr, dz)
                          + V * psi + eos.mu(rho) * psi)
            mu_cur = integrate_cyl((psi.conj() * Hpsi_check).real,
                                   r, z, dr, dz) / N_part
            dmu = abs(mu_cur - mu_prev)
            if it % 5000 == 0:
                r2 = moment_r2(rho, R, r, z, dr, dz, N_part)
                z2 = moment_z2(rho, Z, r, z, dr, dz, N_part)
                elapsed = time_mod.time() - t_start
                print(f"    iter={it:6d}  mu={mu_cur:.6f}  dmu={dmu:.2e}  "
                      f"rho_max={rho.max():.4f}  <r2>={r2:.4f}  <z2>={z2:.4f}  "
                      f"elapsed={elapsed:.0f}s")
            if dmu < tol and it > 500:
                elapsed = time_mod.time() - t_start
                print(f"  Converged at iter {it}: mu={mu_cur:.6f}, dmu={dmu:.2e} "
                      f"({elapsed:.0f}s)")
                break
            mu_prev = mu_cur

    rho = np.abs(psi)**2
    N_final = integrate_cyl(rho, r, z, dr, dz)
    r2 = moment_r2(rho, R, r, z, dr, dz, N_part)
    z2 = moment_z2(rho, Z, r, z, dr, dz, N_part)
    lap_psi = laplacian_cyl(psi, r, dr, dz)
    E_kin = -0.5 * integrate_cyl((psi.conj() * lap_psi).real, r, z, dr, dz)
    E_pot = integrate_cyl(rho * V, r, z, dr, dz)
    E_int = integrate_cyl(rho * eos.eps(rho), r, z, dr, dz)

    Hpsi = (-0.5 * laplacian_cyl(psi, r, dr, dz) + V * psi
            + eos.mu(rho) * psi)
    mu_final = integrate_cyl((psi.conj() * Hpsi).real, r, z, dr, dz) / N_part

    print(f"  Ground state: N={N_final:.6f}, mu={mu_final:.6f}")
    print(f"  rho_peak={rho.max():.4f}")
    print(f"  <r2>={r2:.4f}, <z2>={z2:.4f}")
    print(f"  E_kin={E_kin:.4f}, E_pot={E_pot:.4f}, E_int={E_int:.4f}, "
          f"E_tot={E_kin+E_pot+E_int:.4f}")
    return psi


def load_eos_extended(eos_file, model='poly3'):
    """Load EOS, supporting both poly3_coeffs and poly2_coeffs formats.

    Parameters
    ----------
    model : str
        'poly3' (default) or 'poly2'. Selects which fit to use.
    """
    d = np.load(eos_file, allow_pickle=True)
    if model == 'poly2':
        if 'poly2_coeffs' in d:
            c = d['poly2_coeffs']
            return np.array([c[0], c[1], 0.0])  # pad to 3 coeffs (a3=0)
        else:
            raise ValueError(f"No poly2_coeffs in {eos_file}")
    else:  # poly3
        if 'poly3_coeffs' in d:
            coeffs = d['poly3_coeffs']
            if len(coeffs) == 3:
                return coeffs
            else:
                return np.array([coeffs[0], coeffs[1], 0.0])
        elif 'poly2_coeffs' in d:
            c = d['poly2_coeffs']
            return np.array([c[0], c[1], 0.0])
        else:
            raise ValueError(f"No recognized EOS coefficients in {eos_file}")


def run_N128(args):
    print("=" * 70)
    print(f"Hydrodynamic PDE Solver for N={args.N_part} trap quench")
    print("=" * 70)

    coeffs = load_eos_extended(args.eos_file, model=args.eos_model)
    eos = EOS(coeffs)
    print(f"\nEOS: a1={eos.a1:.6f}, a2={eos.a2:.6f}, a3={eos.a3:.6f}")
    print(f"  rho_stab={eos.rho_stab:.3f}, rho_reg={eos.rho_reg:.3f}")

    N_part = args.N_part
    wr0, wz0 = 1.0, 1.0
    wr1, wz1 = args.omega_rho_quench, 1.0

    r, z, dr, dz, R, Z = make_grid(args.N_r, args.N_z, args.R_max, args.Z_max)
    print(f"\nGrid: {args.N_r}x{args.N_z}, dr={dr:.5f}, dz={dz:.5f}")
    print(f"Domain: r=[0,{args.R_max}], z=[{-args.Z_max},{args.Z_max}]")

    V_pre = 0.5*(wr0**2*R**2 + wz0**2*Z**2)
    V_post = 0.5*(wr1**2*R**2 + wz1**2*Z**2)

    # Initial condition
    if args.ic == 'gaussian':
        # Match tVMC moments
        try:
            tvmc_tmp = np.load(args.tvmc_file, allow_pickle=True)
            rho2_target = float(tvmc_tmp['rho2_gs'])
            z2_target = float(tvmc_tmp['z2_gs'])
            print(f"\nGaussian IC matched to tVMC: <r2>={rho2_target:.4f}, <z2>={z2_target:.4f}")
        except Exception:
            rho2_target = 1.31
            z2_target = 0.65
            print(f"\nGaussian IC with default moments: <r2>={rho2_target}, <z2>={z2_target}")
        sr2, sz2 = rho2_target / 2.0, z2_target
        rho0 = np.exp(-R**2/(2*sr2) - Z**2/(2*sz2))
        rho0 *= N_part / integrate_cyl(rho0, r, z, dr, dz)
        psi = np.sqrt(rho0).astype(complex)
        r2_check = moment_r2(rho0, R, r, z, dr, dz, N_part)
        z2_check = moment_z2(rho0, Z, r, z, dr, dz, N_part)
        print(f"  <r2>={r2_check:.4f}, <z2>={z2_check:.4f}, rho_peak={rho0.max():.4f}")
    else:
        print(f"\nComputing GPE ground state for N={N_part}...")
        t0 = time_mod.time()
        psi = groundstate_ic_N128(r, z, dr, dz, R, Z, N_part, V_pre, eos)
        print(f"Ground state took {time_mod.time()-t0:.0f}s")

    rho0 = np.abs(psi)**2
    N0 = integrate_cyl(rho0, r, z, dr, dz)
    r2_0 = moment_r2(rho0, R, r, z, dr, dz, N_part)
    z2_0 = moment_z2(rho0, Z, r, z, dr, dz, N_part)
    E_pre = gpe_energy(psi, r, z, dr, dz, V_pre, eos, True)
    E_post = gpe_energy(psi, r, z, dr, dz, V_post, eos, True)

    print(f"\nE_pre={E_pre:.4f}, E_post={E_post:.4f}, dE={E_post-E_pre:.4f}")
    print(f"Expected dE = {0.5*(wr1**2-wr0**2)*r2_0*N_part:.4f}")

    # Load tVMC reference
    tvmc = None
    try:
        tvmc = np.load(args.tvmc_file, allow_pickle=True)
        print(f"\ntVMC reference: E_gs={tvmc['E_gs']:.4f}, "
              f"<r2>={tvmc['rho2_gs']:.4f}, <z2>={tvmc['z2_gs']:.4f}")
        print(f"  rho2 range: [{tvmc['rho2_values'].min():.4f}, "
              f"{tvmc['rho2_values'].max():.4f}]")
        print(f"  z2 range: [{tvmc['z2_values'].min():.4f}, "
              f"{tvmc['z2_values'].max():.4f}]")
    except Exception as e:
        print(f"Warning: could not load tVMC file: {e}")

    # Time evolution
    print(f"\nt_max={args.t_max}, CFL={args.cfl}")
    sys.stdout.flush()

    t_cur = 0.0
    step = 0
    ts = [0.0]; r2s = [r2_0]; z2s = [z2_0]; Ns = [N0]; Es = [E_post]
    last_p = -1.0
    V = V_post

    snap_interval = args.t_max / 10
    next_snap = snap_interval
    snapshots = [(0.0, rho0.copy())]
    t_wall_start = time_mod.time()

    while t_cur < args.t_max:
        dt = gpe_cfl(psi, r, dr, dz, eos, args.cfl, True)
        dt = min(dt, args.t_max - t_cur, 0.01)

        psi = gpe_rk4(psi, r, dr, dz, V, eos, dt, True)
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
            Et = gpe_energy(psi, r, z, dr, dz, V, eos, True)
            ts.append(t_cur); r2s.append(r2t)
            z2s.append(z2t); Ns.append(Nt); Es.append(Et)

        if t_cur >= next_snap:
            rho = np.abs(psi)**2
            snapshots.append((t_cur, rho.copy()))
            next_snap += snap_interval

        if t_cur - last_p >= 0.2:
            rho = np.abs(psi)**2
            Nt = integrate_cyl(rho, r, z, dr, dz)
            r2t = moment_r2(rho, R, r, z, dr, dz, N_part)
            z2t = moment_z2(rho, Z, r, z, dr, dz, N_part)
            Et = gpe_energy(psi, r, z, dr, dz, V, eos, True)
            dN = abs(Nt - N0) / N0
            dE = abs(Et - E_post) / max(abs(E_post), 1e-10)
            wall = time_mod.time() - t_wall_start
            print(f"  t={t_cur:.3f}  <r2>={r2t:.4f}  <z2>={z2t:.4f}  "
                  f"rho_max={rho.max():.3f}  dt={dt:.6f}  "
                  f"dN={dN:.2e}  dE={dE:.2e}  step={step}  wall={wall:.0f}s")
            sys.stdout.flush()
            last_p = t_cur

    rho = np.abs(psi)**2
    snapshots.append((t_cur, rho.copy()))

    wall_total = time_mod.time() - t_wall_start
    print(f"\nDone: {step} steps, {wall_total:.0f}s wall time")

    ts = np.array(ts); r2s = np.array(r2s); z2s = np.array(z2s)
    Ns = np.array(Ns); Es = np.array(Es)

    dN_max = np.max(np.abs(Ns - N0) / N0)
    dE_max = np.max(np.abs(Es - E_post) / max(abs(E_post), 1e-10))
    print(f"\nConservation: max |dN/N| = {dN_max:.2e}, max |dE/E| = {dE_max:.2e}")

    # Scaling ansatz
    print("\nScaling ansatz...")
    t_sc, r2_sc, z2_sc = solve_scaling(wr0, wz0, wr1, wz1, r2_0, z2_0,
                                       args.t_max)

    # Save
    snap_times = np.array([s[0] for s in snapshots])
    snap_densities = np.array([s[1] for s in snapshots])
    np.savez(args.output,
             times=ts, rho2=r2s, z2=z2s, energy=Es, particle_number=Ns,
             rho2_0=r2_0, z2_0=z2_0, E_pre=E_pre, E_post=E_post,
             snapshot_times=snap_times, snapshot_densities=snap_densities,
             r_grid=r, z_grid=z,
             scaling_times=t_sc, scaling_rho2=r2_sc, scaling_z2=z2_sc,
             coeffs=coeffs, N_part=N_part,
             N_r=args.N_r, N_z=args.N_z,
             R_max=args.R_max, Z_max=args.Z_max,
             t_max=args.t_max, cfl=args.cfl,
             omega_rho_quench=wr1)
    print(f"Saved to {args.output}")

    # Plot
    if not args.no_plot:
        make_comparison_plot(args, ts, r2s, z2s, Es, Ns, t_sc, r2_sc, z2_sc,
                             E_post, tvmc)

    return ts, r2s, z2s, Es


def make_comparison_plot(args, times, rho2, z2, energy, N_arr,
                         t_sc, rho2_sc, z2_sc, E0_post, tvmc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    ax.plot(times, rho2, 'b-', lw=2, label='Hydro PDE')
    ax.plot(t_sc, rho2_sc, 'g--', lw=1.5, label='Scaling ansatz')
    if tvmc is not None:
        ax.plot(tvmc['times'], tvmc['rho2_values'], 'r-', lw=1, alpha=0.7,
                label='tVMC')
    ax.set_ylabel(r'$\langle \rho^2 \rangle$')
    ax.set_title(f'N={args.N_part} breathing: '
                 r'$\omega_\rho: 1 \to$' + f' {args.omega_rho_quench}')
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
    outpng = args.output.replace('.npz', '.png')
    plt.savefig(outpng, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {outpng}")
    plt.close()


def main():
    p = argparse.ArgumentParser(description='GPE solver for N=128 quench')
    p.add_argument('--eos_file', default='eos/eos_fit_extended.npz')
    p.add_argument('--tvmc_file', default='data/tvmc_N128_new/results.npz')
    p.add_argument('--N_part', type=int, default=128)
    p.add_argument('--omega_rho_quench', type=float, default=2.0)
    p.add_argument('--t_max', type=float, default=3.5)
    p.add_argument('--N_r', type=int, default=192)
    p.add_argument('--N_z', type=int, default=512)
    p.add_argument('--R_max', type=float, default=6.0)
    p.add_argument('--Z_max', type=float, default=8.0)
    p.add_argument('--cfl', type=float, default=0.3)
    p.add_argument('--output', default='hydro_results_N128.npz')
    p.add_argument('--ic', choices=['gs', 'gaussian'], default='gs',
                   help='Initial condition: gs (GPE ground state) or gaussian (matched to tVMC moments)')
    p.add_argument('--eos_model', choices=['poly2', 'poly3'], default='poly3',
                   help='EOS model: poly2 (unconditionally stable) or poly3 (default)')
    p.add_argument('--no_plot', action='store_true')
    args = p.parse_args()
    run_N128(args)


if __name__ == '__main__':
    main()
