#!/usr/bin/env python3
"""
Fig 1 (single-column, vertical):
  (a) Leading-order HNC pair correlation e^{2u2} at θ=0
  (b) Full vs frozen-u2 tVMC dynamics for N=128 (ρ² and z²)
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

# ── Panel (a): exp(2u2) at θ=0 ──
d8 = np.load(f'{BASEDIR}/data/tvmc_N8_freeze_u2/results.npz', allow_pickle=True)
n_rho = int(d8['n_rho'])
n_z = int(d8['n_z'])
n2 = int(d8['n2'])
kappa = float(d8['kappa_cusp'])
eps_c = float(d8['eps_cusp'])
rc_cusp = float(d8['rc_cusp'])
ell2 = np.exp(np.linspace(np.log(float(d8['ell2_min'])),
                           np.log(float(d8['ell2_max'])), n2))

params = d8['params_gs']
idx_c0 = 2 + n_rho + n_z
c0 = params[idx_c0:idx_c0+n2].real
c2 = params[idx_c0+n2:idx_c0+2*n2].real

r = np.linspace(0.001, 2.5, 500)

# Cusp function
cusp = -kappa / np.sqrt(r**2 + eps_c**2) * np.exp(-r**2 / (2*rc_cusp**2))

# f0(r) = cusp + Σ c0_k * exp(-r²/(2ℓ²))
f0 = cusp.copy()
for k in range(n2):
    f0 += c0[k] * np.exp(-r**2 / (2*ell2[k]**2))

# f2(r) = Σ c2_k * r² * exp(-r²/(2ℓ²))
f2 = np.zeros_like(r)
for k in range(n2):
    f2 += c2[k] * r**2 * np.exp(-r**2 / (2*ell2[k]**2))

# u2 at θ=0: P2(cos0) = P2(1) = 1
u2_theta0 = f0 + f2 * 1.0
hnc_theta0 = np.exp(2 * u2_theta0)

# ── Panel (b): Full vs frozen dynamics for N=128 ──
d128_full = np.load(f'{BASEDIR}/data/tvmc_N128/results.npz', allow_pickle=True)
d128_frz = np.load(f'{BASEDIR}/data/tvmc_N128_freeze_u2/results.npz', allow_pickle=True)

# ── Plot ──
fig, axes = plt.subplots(2, 1, figsize=(COL_W, 4.6))

# Panel (a)
ax = axes[0]
ax.plot(r, hnc_theta0, 'b-', lw=1.2)
ax.axhline(1, color='gray', ls=':', lw=0.6)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$e^{2u_2(r,\,\theta{=}0)}$')
ax.set_xlim(0, 2.5)
ax.set_ylim(-0.05, 1.15)
ax.text(0.04, 0.92, r'(a)', transform=ax.transAxes, fontsize=10,
        fontweight='bold', va='top')

# Panel (b)
ax = axes[1]
t_end = min(d128_full['times'][-1], d128_frz['times'][-1])
ax.plot(d128_full['times'], d128_full['rho2_values'], 'r-', lw=0.8, alpha=0.85,
        label=r'full tVMC')
ax.plot(d128_frz['times'], d128_frz['rho2_values'], 'b--', lw=0.8, alpha=0.85,
        label=r'frozen $u_2$')
ax.plot(d128_full['times'], d128_full['z2_values'], 'r-', lw=0.8, alpha=0.5)
ax.plot(d128_frz['times'], d128_frz['z2_values'], 'b--', lw=0.8, alpha=0.5)
# Text labels offset from the curves
ax.annotate(r'$\langle\rho^2\rangle$', xy=(0.55, 1.55), fontsize=8, color='k')
ax.annotate(r'$\langle z^2\rangle$', xy=(0.55, 0.45), fontsize=8, color='k')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\langle r^2\rangle$')
ax.legend(fontsize=7, ncol=1, loc='upper right',
          handlelength=1.5, handletextpad=0.4)
ax.set_xlim(0, t_end)
ax.text(0.04, 0.92, r'(b)', transform=ax.transAxes, fontsize=10,
        fontweight='bold', va='top')

plt.tight_layout(h_pad=1.5)
plt.savefig('fig1_hnc_dynamics.pdf', dpi=300, bbox_inches='tight')
print('Saved fig1_hnc_dynamics.pdf')
plt.close()
