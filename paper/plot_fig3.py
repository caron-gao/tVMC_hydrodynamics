#!/usr/bin/env python3
"""
Fig 3 (single-column): μ(ρ) = d(ρε)/dρ for both interaction strengths
       Showing VMC data points (numerical derivative) and polynomial fits.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=9)

BASEDIR = '..'
COL_W = 3.4  # PRL single-column width in inches


def mu_poly2(rho, a1, a2):
    return 2 * a1 * rho + 3 * a2 * rho**2


# ── Load EOS data ──

# Original interaction C3=1e-3
eos_orig_vmc = np.load(f'{BASEDIR}/eos/eos_vmc_results_extended.npz', allow_pickle=True)
rho_orig = eos_orig_vmc['rho']
E_orig = eos_orig_vmc['E_per_N']
E_orig_err = eos_orig_vmc['E_per_N_err']

eos_orig_fit = np.load(f'{BASEDIR}/eos/eos_fit_extended.npz', allow_pickle=True)
poly2_orig = eos_orig_fit['poly2_coeffs']

# Weak interaction C3=1e-4
eos_weak_vmc = np.load(f'{BASEDIR}/eos/eos_vmc_weak.npz', allow_pickle=True)
rho_weak = eos_weak_vmc['rho']
E_weak = eos_weak_vmc['E_per_N']
E_weak_err = eos_weak_vmc['E_per_N_err']

eos_weak_fit = np.load(f'{BASEDIR}/eos/eos_fit_weak.npz', allow_pickle=True)
poly2_weak = eos_weak_fit['poly2_coeffs']

# Compute μ data points from VMC: μ = d(ρε)/dρ via numerical derivative
def compute_mu_data(rho, eps, eps_err):
    """Compute μ = d(ρε)/dρ from discrete (ρ, ε) data using central differences."""
    rho_eps = rho * eps
    mu = np.gradient(rho_eps, rho)
    # Error propagation (approximate): δ(ρε) ≈ ρ·δε, then δμ ≈ ρ·δε / Δρ
    # Use average spacing for error estimate
    drho = np.gradient(rho)
    mu_err = rho * eps_err / np.abs(drho) * np.sqrt(2)  # factor for diff of noisy data
    # Cap unreasonable errors (at small ρ where spacing is tiny)
    mu_err = np.minimum(mu_err, np.abs(mu) * 0.5)
    return mu, mu_err

mu_orig_data, mu_orig_err = compute_mu_data(rho_orig, E_orig, E_orig_err)
mu_weak_data, mu_weak_err = compute_mu_data(rho_weak, E_weak, E_weak_err)

# Fine grids for fit lines — extend to max data x-value
rho_max_orig = rho_orig.max()
rho_max_weak = rho_weak.max()
rho_fine_orig = np.linspace(0.01, rho_max_orig, 300)
rho_fine_weak = np.linspace(0.01, rho_max_weak, 300)

# Use common x-axis range (up to the larger of the two)
x_max = max(rho_max_orig, rho_max_weak)

# ── Plot ──
fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.6))

# Original C3=1e-3
ax.errorbar(rho_orig, mu_orig_data, yerr=mu_orig_err, fmt='ko', ms=2.5,
            capsize=1.2, lw=0.6, elinewidth=0.5, zorder=5,
            label=r'VMC ($C_3{=}10^{-3}$)')
ax.plot(rho_fine_orig, mu_poly2(rho_fine_orig, *poly2_orig), 'k-', lw=1.2,
        alpha=0.8, label=r'Fit ($C_3{=}10^{-3}$)')

# Weak C3=1e-4
ax.errorbar(rho_weak, mu_weak_data, yerr=mu_weak_err, fmt='rs', ms=2.5,
            capsize=1.2, lw=0.6, elinewidth=0.5, zorder=5,
            label=r'VMC ($C_3{=}10^{-4}$)')
ax.plot(rho_fine_weak, mu_poly2(rho_fine_weak, *poly2_weak), 'r-', lw=1.2,
        alpha=0.8, label=r'Fit ($C_3{=}10^{-4}$)')

ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\mu(\rho) = d(\rho\varepsilon)/d\rho$')
ax.set_xlim(0, 15)
ax.set_ylim(0, 5)

plt.tight_layout()
plt.savefig('fig3_eos.pdf', dpi=300, bbox_inches='tight')
print('Saved fig3_eos.pdf')
plt.close()
