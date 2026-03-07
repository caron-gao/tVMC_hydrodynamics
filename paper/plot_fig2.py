#!/usr/bin/env python3
"""
Fig 2 (figure*, two-column): Δexp(2u2) heatmap at 4 time snapshots
    1×4 panels, r ∈ [0, 0.5], cosθ ∈ [-1, 1]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=9)

BASEDIR = '..'

# Load N=128 full tVMC (has time-dependent u2 snapshots)
d = np.load(f'{BASEDIR}/data/tvmc_N128/results.npz', allow_pickle=True)
n_rho = int(d['n_rho'])
n_z = int(d['n_z'])
n2 = int(d['n2'])
kappa = float(d['kappa_cusp'])
eps_c = float(d['eps_cusp'])
rc_cusp = float(d['rc_cusp'])
ell2 = np.exp(np.linspace(np.log(float(d['ell2_min'])),
                        np.log(float(d['ell2_max'])), n2))

snapshots = d['param_snapshots']
snap_times = d['snapshot_times']
idx_c0 = 2 + n_rho + n_z


def compute_u2_grid(params, r_arr, cos_arr):
    """Compute u2(r, cosθ) on a 2D grid."""
    c0 = params[idx_c0:idx_c0+n2].real
    c2 = params[idx_c0+n2:idx_c0+2*n2].real

    u2 = np.zeros((len(cos_arr), len(r_arr)))
    for ir, r in enumerate(r_arr):
        cusp = -kappa / np.sqrt(r**2 + eps_c**2) * np.exp(-r**2 / (2*rc_cusp**2))
        f0 = cusp
        f2 = 0.0
        for k in range(n2):
            g0 = np.exp(-r**2 / (2*ell2[k]**2))
            f0 += c0[k] * g0
            f2 += c2[k] * r**2 * g0
        for ic, c in enumerate(cos_arr):
            P2 = 0.5 * (3*c**2 - 1)
            u2[ic, ir] = f0 + f2 * P2
    return u2


# Grid (focus on r ∈ [0, 0.5])
r_arr = np.linspace(0.005, 0.5, 200)
cos_arr = np.linspace(-1, 1, 100)

# Select 4 time snapshots close to t=0, 1, 2, 3
target_times = [0.0, 1.0, 2.0, 3.0]
snap_indices = []
for t_target in target_times:
    idx = np.argmin(np.abs(snap_times - t_target))
    snap_indices.append(idx)
    print(f"Target t={t_target:.1f}, actual t={snap_times[idx]:.3f}, index={idx}")

# Compute u2 grids and exp(2u2)
u2_grids = []
g2_grids = []
for si in snap_indices:
    u2 = compute_u2_grid(snapshots[si], r_arr, cos_arr)
    u2_grids.append(u2)
    g2_grids.append(np.exp(2 * u2))

g2_0 = g2_grids[0]

# Compute Δexp(2u2) = exp(2u2(t)) - exp(2u2(0))
dg2_grids = [g2 - g2_0 for g2 in g2_grids]

# Print diagnostics
for col in range(1, 4):
    dg = dg2_grids[col]
    print(f"t={snap_times[snap_indices[col]]:.2f}: "
        f"Δg2 range [{dg.min():.4f}, {dg.max():.4f}]")

# ── Plot ──
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.6))

vmin, vmax = -0.03, 0.03
labels = ['(a)', '(b)', '(c)', '(d)']

for col, (si, t_target) in enumerate(zip(snap_indices, target_times)):
    t_label = f"$t={snap_times[si]:.1f}$"

    ax = axes[col]
    im = ax.pcolormesh(r_arr, cos_arr, dg2_grids[col], cmap='RdBu_r',
                    vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(t_label, fontsize=9)
    ax.set_xlabel(r'$r$')
    if col == 0:
        ax.set_ylabel(r'$\cos\theta$')
    else:
        ax.set_yticklabels([])
    ax.text(0.06, 0.92, labels[col], transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', color='k',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

# Colorbar
fig.subplots_adjust(right=0.88, wspace=0.08)
cax = fig.add_axes([0.90, 0.18, 0.015, 0.65])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r'$\Delta\, e^{2u_2}$', fontsize=9)
cb.ax.tick_params(labelsize=7)

plt.savefig('fig2_u2_heatmap.pdf', dpi=600, bbox_inches='tight')
print('Saved fig2_u2_heatmap.pdf')
plt.close()
