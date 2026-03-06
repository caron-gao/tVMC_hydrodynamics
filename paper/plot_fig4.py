#!/usr/bin/env python3
"""
Fig 4 (single-column, vertical):
  (a) N=128 GPE dynamics vs tVMC (validation)
  (b) N=8192 GPE prediction (same C3=1e-3 interaction)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=9)

BASEDIR = '..'
COL_W = 3.4

fig, axes = plt.subplots(2, 1, figsize=(COL_W, 4.6))

# ── Panel (a): N=128 GPE vs tVMC ──
ax = axes[0]

# tVMC reference (frozen-u2)
d128 = np.load(f'{BASEDIR}/data/tvmc_N128_freeze_u2/results.npz', allow_pickle=True)
t_tvmc = d128['times']
rho2_tvmc = d128['rho2_values']
z2_tvmc = d128['z2_values']

# GPE result
gpe_candidates = [
    f'{BASEDIR}/hydro/results/hydro_results_N128_new_gs_poly2.npz',
    f'{BASEDIR}/hydro/results/hydro_results_N128_gs_poly2.npz',
    f'{BASEDIR}/hydro/results/hydro_results_N128.npz',
]
gpe_file = None
for f in gpe_candidates:
    if os.path.exists(f):
        gpe_file = f
        break

ax.plot(t_tvmc, rho2_tvmc, 'r-', lw=0.7, alpha=0.8, label=r'tVMC')
ax.plot(t_tvmc, z2_tvmc, 'r-', lw=0.7, alpha=0.5)
if gpe_file:
    dgpe = np.load(gpe_file, allow_pickle=True)
    t_gpe = dgpe['times']
    rho2_gpe = dgpe['rho2']
    z2_gpe = dgpe['z2']
    ax.plot(t_gpe, rho2_gpe, 'b--', lw=0.8, label=r'GPE')
    ax.plot(t_gpe, z2_gpe, 'b--', lw=0.8, alpha=0.6)
    print(f"Loaded GPE: {gpe_file}")
else:
    print("WARNING: No N=128 GPE result found!")

ax.annotate(r'$\langle\rho^2\rangle$', xy=(1.5, 1.4), fontsize=8)
ax.annotate(r'$\langle z^2\rangle$', xy=(1.5, 0.7), fontsize=8)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\langle r^2\rangle$')
ax.legend(fontsize=7, loc='upper right',
          handlelength=1.5, handletextpad=0.4)
ax.set_xlim(0, min(t_tvmc[-1], 6.0))
ax.text(0.04, 0.92, r'(a)', transform=ax.transAxes, fontsize=10,
        fontweight='bold', va='top')

# ── Panel (b): N=8192 prediction (same interaction C3=1e-3) ──
ax = axes[1]

d8k = np.load(f'{BASEDIR}/hydro/results/hydro_results_N8192.npz',
              allow_pickle=True)
t_8k = d8k['times']
rho2_8k = d8k['rho2']
z2_8k = d8k['z2']

ax.plot(t_8k, rho2_8k, 'b-', lw=0.7)
ax.plot(t_8k, z2_8k, 'r-', lw=0.7)

ax.annotate(r'$\langle\rho^2\rangle$', xy=(1.5, 6.5), fontsize=8)
ax.annotate(r'$\langle z^2\rangle$', xy=(1.5, 3.5), fontsize=8)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\langle r^2\rangle$')
ax.set_xlim(0, t_8k[-1])
ax.text(0.04, 0.92, r'(b)', transform=ax.transAxes, fontsize=10,
        fontweight='bold', va='top')

plt.tight_layout(h_pad=1.5)
plt.savefig('fig4_dynamics.pdf', dpi=300, bbox_inches='tight')
print('Saved fig4_dynamics.pdf')
plt.close()
