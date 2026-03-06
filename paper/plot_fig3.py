#!/usr/bin/env python3
"""
Fig 3 (single-column, flat): μ(ρ) from improved VMC for C3=1e-3.
       Shows VMC data points (numerical derivative) and polynomial fit.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=9)

BASEDIR = '..'
COL_W = 3.4


def mu_poly2(rho, a1, a2):
    return 2 * a1 * rho + 3 * a2 * rho**2


# ── Load improved EOS data (C3=1e-3, high quality) ──
import os
hq_vmc = f'{BASEDIR}/eos/eos_vmc_C3_1e-3_hq.npz'
hq_fit = f'{BASEDIR}/eos/eos_fit_C3_1e-3_hq.npz'

if os.path.exists(hq_vmc):
    eos_vmc = np.load(hq_vmc, allow_pickle=True)
    print(f"Loaded HQ VMC: {hq_vmc}")
else:
    eos_vmc = np.load(f'{BASEDIR}/eos/eos_vmc_results_extended.npz', allow_pickle=True)
    print("Fallback to extended VMC")

if os.path.exists(hq_fit):
    eos_fit = np.load(hq_fit, allow_pickle=True)
    print(f"Loaded HQ fit: {hq_fit}")
else:
    eos_fit = np.load(f'{BASEDIR}/eos/eos_fit_extended.npz', allow_pickle=True)
    print("Fallback to extended fit")

rho_data = eos_vmc['rho']
E_data = eos_vmc['E_per_N']
E_err = eos_vmc['E_per_N_err']
poly2_coeffs = eos_fit['poly2_coeffs']

print(f"poly2 coeffs: a1={poly2_coeffs[0]:.6f}, a2={poly2_coeffs[1]:.6f}")

# Compute μ data points: μ = d(ρε)/dρ via numerical derivative
rho_eps = rho_data * E_data
mu_data = np.gradient(rho_eps, rho_data)
drho = np.gradient(rho_data)
mu_err = rho_data * E_err / np.abs(drho) * np.sqrt(2)
mu_err = np.minimum(mu_err, np.abs(mu_data) * 0.5)

# Fit line extending to max data range
rho_fine = np.linspace(0.01, rho_data.max(), 300)

# ── Plot ──
fig, ax = plt.subplots(1, 1, figsize=(COL_W, 1.8))

ax.errorbar(rho_data, mu_data, yerr=mu_err, fmt='ko', ms=2.5,
            capsize=1.2, lw=0.6, elinewidth=0.5, zorder=5)
ax.plot(rho_fine, mu_poly2(rho_fine, *poly2_coeffs), 'r-', lw=1.2, alpha=0.8)

ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\mu(\rho)$')
ax.set_xlim(0, 15)
ax.set_ylim(0, 5)

plt.tight_layout()
plt.savefig('fig3_eos.pdf', dpi=300, bbox_inches='tight')
print('Saved fig3_eos.pdf')
plt.close()
