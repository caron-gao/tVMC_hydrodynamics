#!/usr/bin/env python3
"""
GPE solver for large-N (N=1000-10000) trap quench dynamics.

Demonstrates the power of the reduced hydrodynamic theory:
the GPE solver cost is independent of N (grid-based PDE),
so simulating N=10000 costs the same as N=8.

Uses EOS from frozen-u2 VMC with configurable interaction parameters.
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
import sys
import time as time_mod

# Import core routines from hydro_solver
from hydro_solver import (
    EOS, load_eos, make_grid, integrate_cyl, moment_r2, moment_z2,
    laplacian_cyl, grad_sq,
    gpe_rhs, gpe_energy, gpe_rk4, gpe_cfl,
    solve_scaling,
)


def load_eos_extended(eos_file, model='poly2'):
    """Load EOS, supporting both poly3_coeffs and poly2_coeffs formats."""
    d = np.load(eos_file, allow_pickle=True)
    if model == 'poly2':
        if 'poly2_coeffs' in d:
            c = d['poly2_coeffs']
            return np.array([c[0], c[1], 0.0])
        elif 'coefficients' in d:
            c = d['coefficients']
            if len(c) == 2:
                return np.array([c[0], c[1], 0.0])
            return c[:3] if len(c) >= 3 else np.array([c[0], c[1] if len(c)>1 else 0.0, 0.0])
        else:
            raise ValueError(f"No poly2_coeffs in {eos_file}")
    else:  # poly3
        if 'poly3_coeffs' in d:
            return d['poly3_coeffs'][:3]
        elif 'coefficients' in d:
            c = d['coefficients']
            if len(c) == 3:
                return c
            return np.array([c[0], c[1], 0.0])
        else:
            raise ValueError(f"No recognized EOS coefficients in {eos_file}")


def estimate_domain(N_part, eos):
    """Estimate cloud size and appropriate grid domain for given N.

    In the Thomas-Fermi limit, the cloud radius scales as:
        R_TF ~ (15*N*g/(4*pi))^(1/5) for isotropic 3D trap (omega=1)
    where g = 2*a1 is the effective coupling from EOS.

    For weakly interacting systems, the cloud is closer to the
    non-interacting size sigma = 1/sqrt(omega) = 1.
    """
    g_eff = 2 * eos.a1  # leading-order coupling
    # TF chemical potential: mu = (15*N*g/(8*pi*sqrt(2)))^(2/5) * omega/2
    # But simpler: R_TF^2 = 2*mu/omega^2, mu = g*rho_0, rho_0 = mu/g
    # So R_TF = (15*N*g/(4*pi))^(1/5)
    R_TF = (15 * N_part * g_eff / (4 * np.pi)) ** 0.2
    mu_TF = 0.5 * R_TF**2  # omega=1

    # Non-interacting size for reference
    sigma_ho = 1.0  # sqrt(hbar/(m*omega))

    # Use max of TF and non-interacting, with safety margin
    R_cloud = max(R_TF, 2.0 * sigma_ho)
    R_max = R_cloud * 2.5  # plenty of room
    Z_max = R_max  # isotropic initially

    # Peak density estimate
    rho_peak_est = mu_TF / g_eff if g_eff > 1e-10 else N_part / (4*np.pi/3 * sigma_ho**3)

    print(f"  Domain estimation for N={N_part}:")
    print(f"    g_eff = 2*a1 = {g_eff:.6f}")
    print(f"    R_TF = {R_TF:.2f}, mu_TF = {mu_TF:.2f}")
    print(f"    sigma_ho = {sigma_ho:.2f}")
    print(f"    rho_peak_est = {rho_peak_est:.1f}")
    print(f"    R_max = {R_max:.1f}, Z_max = {Z_max:.1f}")

    return R_max, Z_max, rho_peak_est


def groundstate_largeN(r, z, dr, dz, R, Z, N_part, V, eos,
                       dtau=None, tol=1e-8, max_iter=100000):
    """GPE ground state via imaginary-time propagation for large N."""
    print("  Computing GPE ground state (imaginary-time propagation)...")
    print(f"  N={N_part}, grid={len(r)}x{len(z)}, "
          f"domain=[0,{r[-1]:.1f}]x[{z[0]:.1f},{z[-1]:.1f}]")

    dx = min(dr, dz)
    if dtau is None:
        dtau = 0.3 * dx**2 / np.pi

    # Initial guess: Gaussian with width between HO and TF
    g_eff = 2 * eos.a1
    R_TF = max((15 * N_part * g_eff / (4 * np.pi)) ** 0.2, 1.5)
    sr2 = max(0.3 * R_TF**2, 0.8)  # intermediate width
    sz2 = sr2
    rho0 = np.exp(-R**2/(2*sr2) - Z**2/(2*sz2))
    rho0 *= N_part / integrate_cyl(rho0, r, z, dr, dz)
    psi = np.sqrt(rho0).astype(complex)

    mu_prev = 1e10
    t_start = time_mod.time()
    for it in range(max_iter):
        rho = np.abs(psi)**2

        # GPE RHS (real-time): i dpsi/dt = H psi
        # Imaginary time: dpsi/dtau = -H psi
        lap = laplacian_cyl(psi, r, dr, dz)
        mu_arr = eos.mu(rho)
        H_psi = -0.5 * lap + (V + mu_arr) * psi
        psi = psi - dtau * H_psi

        # Re-normalize
        rho = np.abs(psi)**2
        Nt = integrate_cyl(rho, r, z, dr, dz)
        psi *= np.sqrt(N_part / Nt)

        if (it+1) % 2000 == 0:
            rho = np.abs(psi)**2
            Nt = integrate_cyl(rho, r, z, dr, dz)
            E_kin = -0.5 * np.real(integrate_cyl(np.conj(psi) * lap, r, z, dr, dz))
            E_pot = integrate_cyl(rho * V, r, z, dr, dz)
            E_int = integrate_cyl(rho * eos.eps(rho), r, z, dr, dz)
            mu_est = (E_kin + E_pot + E_int) / N_part
            r2 = moment_r2(rho, R, r, z, dr, dz, N_part)
            z2 = moment_z2(rho, Z, r, z, dr, dz, N_part)
            elapsed = time_mod.time() - t_start
            print(f"    iter {it+1}: mu={mu_est:.6f}, <r2>={r2:.4f}, "
                  f"<z2>={z2:.4f}, rho_max={rho.max():.3f}, "
                  f"E={E_kin+E_pot+E_int:.4f}, wall={elapsed:.0f}s")

            if abs(mu_est - mu_prev) / max(abs(mu_est), 1e-10) < tol:
                print(f"  Converged at iteration {it+1}")
                break
            mu_prev = mu_est

    rho = np.abs(psi)**2
    lap = laplacian_cyl(psi, r, dr, dz)
    E_kin = -0.5 * np.real(integrate_cyl(np.conj(psi) * lap, r, z, dr, dz))
    E_pot = integrate_cyl(rho * V, r, z, dr, dz)
    E_int = integrate_cyl(rho * eos.eps(rho), r, z, dr, dz)
    print(f"  Ground state: E_kin={E_kin:.4f}, E_pot={E_pot:.4f}, "
          f"E_int={E_int:.4f}, E_tot={E_kin+E_pot+E_int:.4f}")
    return psi


def run_largeN(args):
    print("=" * 70)
    print(f"GPE Solver for N={args.N_part} trap quench")
    print(f"  Demonstrating scalability of reduced hydrodynamic theory")
    print("=" * 70)

    coeffs = load_eos_extended(args.eos_file, model=args.eos_model)
    eos = EOS(coeffs)
    print(f"\nEOS ({args.eos_model}): a1={eos.a1:.6f}, a2={eos.a2:.6f}, a3={eos.a3:.6f}")
    print(f"  rho_stab={eos.rho_stab:.3f}, rho_reg={eos.rho_reg:.3f}")

    N_part = args.N_part
    wr0, wz0 = 1.0, 1.0
    wr1, wz1 = args.omega_rho_quench, 1.0

    # Auto-determine domain if not specified
    if args.R_max is None or args.Z_max is None:
        R_auto, Z_auto, rho_peak_est = estimate_domain(N_part, eos)
        if args.R_max is None:
            args.R_max = R_auto
        if args.Z_max is None:
            args.Z_max = Z_auto
    else:
        rho_peak_est = None

    r, z, dr, dz, R, Z = make_grid(args.N_r, args.N_z, args.R_max, args.Z_max)
    print(f"\nGrid: {args.N_r}x{args.N_z}, dr={dr:.5f}, dz={dz:.5f}")
    print(f"Domain: r=[0,{args.R_max:.1f}], z=[{-args.Z_max:.1f},{args.Z_max:.1f}]")

    V_pre = 0.5*(wr0**2*R**2 + wz0**2*Z**2)
    V_post = 0.5*(wr1**2*R**2 + wz1**2*Z**2)

    # Ground state
    print(f"\nComputing GPE ground state for N={N_part}...")
    t0 = time_mod.time()
    psi = groundstate_largeN(r, z, dr, dz, R, Z, N_part, V_pre, eos)
    print(f"Ground state took {time_mod.time()-t0:.0f}s")

    rho0 = np.abs(psi)**2
    N0 = integrate_cyl(rho0, r, z, dr, dz)
    r2_0 = moment_r2(rho0, R, r, z, dr, dz, N_part)
    z2_0 = moment_z2(rho0, Z, r, z, dr, dz, N_part)
    E_pre = gpe_energy(psi, r, z, dr, dz, V_pre, eos, True)
    E_post = gpe_energy(psi, r, z, dr, dz, V_post, eos, True)

    print(f"\nGround state: N={N0:.4f}, <r2>={r2_0:.4f}, <z2>={z2_0:.4f}")
    print(f"  rho_max={rho0.max():.4f}")
    print(f"  E_pre={E_pre:.4f}, E_post={E_post:.4f}, dE={E_post-E_pre:.4f}")
    print(f"  Expected dE = {0.5*(wr1**2-wr0**2)*r2_0*N_part:.4f}")

    # Time evolution
    print(f"\nQuench: omega_rho {wr0} -> {wr1}")
    print(f"t_max={args.t_max}, CFL={args.cfl}")
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
        make_plot(args, ts, r2s, z2s, Es, Ns, t_sc, r2_sc, z2_sc, E_post)

    return ts, r2s, z2s, Es


def make_plot(args, times, rho2, z2, energy, N_arr,
              t_sc, rho2_sc, z2_sc, E0_post):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) rho2(t)
    ax = axes[0, 0]
    ax.plot(times, rho2, 'b-', lw=2, label='GPE (this work)')
    ax.plot(t_sc, rho2_sc, 'g--', lw=1.5, label='Scaling ansatz')
    ax.set_ylabel(r'$\langle \rho^2 \rangle$')
    ax.set_title(f'N={args.N_part}: ' + r'$\omega_\rho: 1 \to$'
                 + f' {args.omega_rho_quench}')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (b) z2(t)
    ax = axes[0, 1]
    ax.plot(times, z2, 'b-', lw=2, label='GPE')
    ax.plot(t_sc, z2_sc, 'g--', lw=1.5, label='Scaling ansatz')
    ax.set_ylabel(r'$\langle z^2 \rangle$')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (c) Conservation
    ax = axes[1, 0]
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

    # (d) FFT of rho2
    ax = axes[1, 1]
    dt_avg = np.mean(np.diff(times))
    from scipy.fft import rfft, rfftfreq
    r2_detrend = rho2 - np.mean(rho2)
    window = np.hanning(len(r2_detrend))
    freqs = rfftfreq(len(r2_detrend), d=dt_avg) * 2 * np.pi
    power = np.abs(rfft(r2_detrend * window))**2
    mask = freqs > 0.5
    ax.plot(freqs[mask], power[mask] / power[mask].max(), 'b-', lw=1.5)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('Power (arb.)')
    ax.set_title(r'FFT of $\langle\rho^2\rangle$')
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3)
    # Mark dominant frequency
    peak_idx = np.argmax(power[mask])
    omega_br = freqs[mask][peak_idx]
    ax.axvline(omega_br, color='r', ls='--', alpha=0.7,
               label=f'$\\omega_{{br}}={omega_br:.2f}$')
    ax.legend()

    plt.tight_layout()
    outpng = args.output.replace('.npz', '.png')
    plt.savefig(outpng, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {outpng}")
    plt.close()


def main():
    p = argparse.ArgumentParser(
        description='GPE solver for large-N quench dynamics')
    p.add_argument('--eos_file', required=True,
                   help='EOS fit file (from fit_eos.py)')
    p.add_argument('--N_part', type=int, required=True,
                   help='Number of particles')
    p.add_argument('--omega_rho_quench', type=float, default=2.0,
                   help='Post-quench radial frequency (default: 2.0)')
    p.add_argument('--t_max', type=float, default=3.5,
                   help='Simulation time (default: 3.5)')
    p.add_argument('--N_r', type=int, default=256,
                   help='Radial grid points (default: 256)')
    p.add_argument('--N_z', type=int, default=512,
                   help='Axial grid points (default: 512)')
    p.add_argument('--R_max', type=float, default=None,
                   help='Radial domain (auto if not set)')
    p.add_argument('--Z_max', type=float, default=None,
                   help='Axial domain (auto if not set)')
    p.add_argument('--cfl', type=float, default=0.3,
                   help='CFL factor (default: 0.3)')
    p.add_argument('--output', default=None,
                   help='Output file (default: hydro_results_N{N_part}.npz)')
    p.add_argument('--eos_model', choices=['poly2', 'poly3'], default='poly2',
                   help='EOS model (default: poly2, unconditionally stable)')
    p.add_argument('--no_plot', action='store_true')
    args = p.parse_args()

    if args.output is None:
        args.output = f'hydro_results_N{args.N_part}.npz'

    run_largeN(args)


if __name__ == '__main__':
    main()
