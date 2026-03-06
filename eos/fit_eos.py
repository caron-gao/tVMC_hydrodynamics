#!/usr/bin/env python3
"""
fit_eos.py
Fit equation of state ε(ρ) from uniform VMC data.
Compute chemical potential μ(ρ) = d(ρε)/dρ for hydro PDE solver.

Usage:
  python fit_eos.py                              # Default input
  python fit_eos.py --input eos_vmc_results.npz  # Custom input
  python fit_eos.py --no_plot                    # Skip plotting

Output: eos_fit.npz + plots
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# Model functions
# ==============================================================================

def eps_poly2(rho, a1, a2):
    """ε(ρ) = a₁ρ + a₂ρ²  (no constant: ε→0 as ρ→0)"""
    return a1 * rho + a2 * rho**2


def eps_poly3(rho, a1, a2, a3):
    """ε(ρ) = a₁ρ + a₂ρ² + a₃ρ³"""
    return a1 * rho + a2 * rho**2 + a3 * rho**3


def mu_from_poly2(rho, a1, a2):
    """μ(ρ) = d(ρε)/dρ = 2a₁ρ + 3a₂ρ²"""
    return 2 * a1 * rho + 3 * a2 * rho**2


def mu_from_poly3(rho, a1, a2, a3):
    """μ(ρ) = d(ρε)/dρ = 2a₁ρ + 3a₂ρ² + 4a₃ρ³"""
    return 2 * a1 * rho + 3 * a2 * rho**2 + 4 * a3 * rho**3


# ==============================================================================
# Error analysis
# ==============================================================================

def blocking_error(data, max_blocks=10):
    """Estimate SEM using blocking analysis (accounts for autocorrelation)."""
    n = len(data)
    errors = []

    for k in range(max_blocks):
        bs = 2 ** k
        n_blocks = n // bs
        if n_blocks < 4:
            break
        block_means = np.array([
            np.mean(data[i * bs:(i + 1) * bs])
            for i in range(n_blocks)
        ])
        sem = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
        errors.append(sem)

    if not errors:
        return np.std(data, ddof=1) / np.sqrt(n)
    return max(errors)  # Conservative: largest block error


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fit EOS from uniform VMC data')
    parser.add_argument('--input', type=str, default='eos_vmc_results.npz',
                        help='Input VMC results file')
    parser.add_argument('--output', type=str, default='eos_fit.npz',
                        help='Output fit results file')
    parser.add_argument('--plot_prefix', type=str, default='eos_',
                        help='Prefix for plot files')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip generating plots')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = np.load(args.input)
    rho = data['rho']
    E_per_N = data['E_per_N']
    E_per_N_err = data['E_per_N_err']
    E_kin_per_N = data['E_kin_per_N']
    E_int_per_N = data['E_int_per_N']
    N = int(data['N'])

    n_rho = len(rho)
    print(f"Loaded {n_rho} density points from {args.input}")
    print(f"  rho range: [{rho[0]:.4f}, {rho[-1]:.4f}]")
    print(f"  E/N range: [{E_per_N[0]:.6f}, {E_per_N[-1]:.6f}]")

    # ------------------------------------------------------------------
    # Blocking analysis for better error estimates
    # ------------------------------------------------------------------
    if 'batch_E' in data:
        batch_E = data['batch_E']
        E_per_N_err_block = np.zeros(n_rho)
        for i in range(n_rho):
            E_per_N_err_block[i] = blocking_error(batch_E[i])
        # Use the larger of naive SEM and blocking error
        E_per_N_err = np.maximum(E_per_N_err, E_per_N_err_block)
        print("  Applied blocking analysis for error estimates")

    if 'E_kin_per_N_err' in data:
        E_kin_err = data['E_kin_per_N_err']
    else:
        E_kin_err = E_per_N_err  # Fallback

    if 'E_int_per_N_err' in data:
        E_int_err = data['E_int_per_N_err']
    else:
        E_int_err = E_per_N_err  # Fallback

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    print("\nSanity checks:")
    print(f"  E/N at lowest rho ({rho[0]:.3f}): {E_per_N[0]:.6f} "
          f"(should be ~0 for rho->0)")

    if np.any(np.isnan(E_per_N)):
        bad = np.where(np.isnan(E_per_N))[0]
        print(f"  WARNING: NaN values at indices {bad}, excluding from fit")
        mask_good = ~np.isnan(E_per_N)
        rho = rho[mask_good]
        E_per_N = E_per_N[mask_good]
        E_per_N_err = E_per_N_err[mask_good]
        E_kin_per_N = E_kin_per_N[mask_good]
        E_int_per_N = E_int_per_N[mask_good]
        E_kin_err = E_kin_err[mask_good]
        E_int_err = E_int_err[mask_good]
        n_rho = len(rho)

    # ------------------------------------------------------------------
    # Weighted polynomial fits
    # ------------------------------------------------------------------
    sigma = np.maximum(E_per_N_err, 1e-10)  # Avoid zero weights

    # Fit poly2: ε = a₁ρ + a₂ρ²
    popt2, pcov2, chi2r_2, aic_2, res2 = None, None, None, None, None
    try:
        popt2, pcov2 = curve_fit(eps_poly2, rho, E_per_N, sigma=sigma,
                                  absolute_sigma=True)
        res2 = E_per_N - eps_poly2(rho, *popt2)
        chi2_2 = np.sum((res2 / sigma) ** 2)
        dof2 = n_rho - 2
        chi2r_2 = chi2_2 / max(dof2, 1)
        aic_2 = chi2_2 + 2 * 2
        perr2 = np.sqrt(np.diag(pcov2))
        print(f"\nPoly2 fit: eps = a1*rho + a2*rho^2")
        print(f"  a1 = {popt2[0]:.6f} +/- {perr2[0]:.6f}")
        print(f"  a2 = {popt2[1]:.6f} +/- {perr2[1]:.6f}")
        print(f"  chi2/dof = {chi2r_2:.3f}, AIC = {aic_2:.1f}")
    except Exception as e:
        print(f"\nPoly2 fit FAILED: {e}")

    # Fit poly3: ε = a₁ρ + a₂ρ² + a₃ρ³
    popt3, pcov3, chi2r_3, aic_3, res3 = None, None, None, None, None
    try:
        popt3, pcov3 = curve_fit(eps_poly3, rho, E_per_N, sigma=sigma,
                                  absolute_sigma=True)
        res3 = E_per_N - eps_poly3(rho, *popt3)
        chi2_3 = np.sum((res3 / sigma) ** 2)
        dof3 = n_rho - 3
        chi2r_3 = chi2_3 / max(dof3, 1)
        aic_3 = chi2_3 + 2 * 3
        perr3 = np.sqrt(np.diag(pcov3))
        print(f"\nPoly3 fit: eps = a1*rho + a2*rho^2 + a3*rho^3")
        print(f"  a1 = {popt3[0]:.6f} +/- {perr3[0]:.6f}")
        print(f"  a2 = {popt3[1]:.6f} +/- {perr3[1]:.6f}")
        print(f"  a3 = {popt3[2]:.6f} +/- {perr3[2]:.6f}")
        print(f"  chi2/dof = {chi2r_3:.3f}, AIC = {aic_3:.1f}")
    except Exception as e:
        print(f"\nPoly3 fit FAILED: {e}")

    # ------------------------------------------------------------------
    # Model selection via AIC
    # ------------------------------------------------------------------
    if popt2 is not None and popt3 is not None:
        delta_aic = aic_2 - aic_3
        if delta_aic > 2:  # poly3 strongly preferred
            print(f"\n=> Selected: poly3 (delta_AIC = {delta_aic:.1f})")
            best = 'poly3'
            popt, pcov = popt3, pcov3
            eps_fn, mu_fn = eps_poly3, mu_from_poly3
            chi2r, residuals = chi2r_3, res3
        else:
            print(f"\n=> Selected: poly2 (delta_AIC = {delta_aic:.1f}, "
                  f"no strong improvement from poly3)")
            best = 'poly2'
            popt, pcov = popt2, pcov2
            eps_fn, mu_fn = eps_poly2, mu_from_poly2
            chi2r, residuals = chi2r_2, res2
    elif popt3 is not None:
        best, popt, pcov = 'poly3', popt3, pcov3
        eps_fn, mu_fn = eps_poly3, mu_from_poly3
        chi2r, residuals = chi2r_3, res3
    elif popt2 is not None:
        best, popt, pcov = 'poly2', popt2, pcov2
        eps_fn, mu_fn = eps_poly2, mu_from_poly2
        chi2r, residuals = chi2r_2, res2
    else:
        print("\nERROR: Both fits failed! Cannot proceed.")
        return

    # ------------------------------------------------------------------
    # Validation: compare with trapped system at central density
    # ------------------------------------------------------------------
    rho0 = 0.67
    eps_pred = eps_fn(rho0, *popt)
    mu_pred = mu_fn(rho0, *popt)
    print(f"\nValidation at trapped central density rho_0 = {rho0}:")
    print(f"  eps(rho_0) = {eps_pred:.6f}  (EOS prediction)")
    print(f"  mu(rho_0)  = {mu_pred:.6f}  (chemical potential)")
    print(f"  Expected E_int/N ~ 0.053 from trapped tVMC")

    # ------------------------------------------------------------------
    # Compute fine grids for output
    # ------------------------------------------------------------------
    rho_fine = np.linspace(0.01, rho[-1] * 1.1, 500)
    eps_fine = eps_fn(rho_fine, *popt)
    mu_fine = mu_fn(rho_fine, *popt)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    save_dict = dict(
        model=best,
        coefficients=popt,
        covariance=pcov,
        chi2_reduced=chi2r,
        rho_valid_min=rho[0],
        rho_valid_max=rho[-1],
        # Fine grids for interpolation
        rho_fine=rho_fine,
        eps_fine=eps_fine,
        mu_fine=mu_fine,
        # Original data
        rho_data=rho,
        E_per_N_data=E_per_N,
        E_per_N_err_data=E_per_N_err,
    )
    # Include both fits if available
    if popt2 is not None:
        save_dict['poly2_coeffs'] = popt2
        save_dict['poly2_cov'] = pcov2
        save_dict['poly2_chi2r'] = chi2r_2
    if popt3 is not None:
        save_dict['poly3_coeffs'] = popt3
        save_dict['poly3_cov'] = pcov3
        save_dict['poly3_chi2r'] = chi2r_3

    np.savez(args.output, **save_dict)
    print(f"\nFit results saved to {args.output}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if args.no_plot:
        print("Skipping plots (--no_plot)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) ε(ρ) with data + fit
    ax = axes[0, 0]
    ax.errorbar(rho, E_per_N, yerr=E_per_N_err, fmt='ko', ms=4,
                capsize=2, label='VMC data')
    ax.plot(rho_fine, eps_fine, 'r-', lw=2,
            label=f'{best} fit')
    if popt2 is not None:
        ax.plot(rho_fine, eps_poly2(rho_fine, *popt2), 'b--', alpha=0.4,
                label='poly2')
    if popt3 is not None and best != 'poly3':
        ax.plot(rho_fine, eps_poly3(rho_fine, *popt3), 'g--', alpha=0.4,
                label='poly3')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\varepsilon(\rho) = E/N$')
    ax.set_title('Energy per particle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) μ(ρ) = chemical potential
    ax = axes[0, 1]
    ax.plot(rho_fine, mu_fine, 'r-', lw=2, label=f'{best}')
    ax.axvline(rho0, color='gray', ls='--', alpha=0.5,
               label=r'$\rho_0 = 0.67$')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\mu(\rho) = d(\rho\varepsilon)/d\rho$')
    ax.set_title('Chemical potential')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Residuals
    ax = axes[1, 0]
    ax.errorbar(rho, residuals, yerr=E_per_N_err, fmt='ko', ms=4, capsize=2)
    ax.axhline(0, color='r', ls='-', alpha=0.5)
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel('Residual (data - fit)')
    ax.set_title(r'Residuals ($\chi^2$/dof = ' + f'{chi2r:.2f})')
    ax.grid(True, alpha=0.3)

    # (d) Kinetic/interaction decomposition
    ax = axes[1, 1]
    ax.errorbar(rho, E_kin_per_N, yerr=E_kin_err, fmt='bs-', ms=4,
                capsize=2, label=r'$E_{kin}/N$')
    ax.errorbar(rho, E_int_per_N, yerr=E_int_err, fmt='r^-', ms=4,
                capsize=2, label=r'$E_{int}/N$')
    ax.errorbar(rho, E_per_N, yerr=E_per_N_err, fmt='ko-', ms=4,
                capsize=2, label=r'$E_{tot}/N$')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel('Energy per particle')
    ax.set_title('Kinetic/Interaction decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f'{args.plot_prefix}fit.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()


if __name__ == '__main__':
    main()
