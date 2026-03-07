#!/usr/bin/env python3
"""
SM figure: Finite-size convergence of EOS coefficient a_1.
Plots a_1(N) vs 1/N with linear extrapolation to N→∞.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=9)

import os
BASEDIR = os.path.join(os.path.dirname(__file__), '..')

# Load data
N_vals = [8, 16, 32]
a1_poly2 = []
a1_poly3 = []

for N in N_vals:
    d = np.load(os.path.join(BASEDIR, 'eos', f'eos_fit_C3_1e-3_N{N}.npz'),
                allow_pickle=True)
    a1_poly2.append(d['poly2_coeffs'][0])
    a1_poly3.append(d['poly3_coeffs'][0])

a1_poly2 = np.array(a1_poly2)
a1_poly3 = np.array(a1_poly3)
inv_N = 1.0 / np.array(N_vals)

# Linear fit: a1 = a1_inf + slope / N
p2 = np.polyfit(inv_N, a1_poly2, 1)
a1_inf_p2 = p2[1]

p3 = np.polyfit(inv_N, a1_poly3, 1)
a1_inf_p3 = p3[1]

# Plot
fig, ax = plt.subplots(figsize=(3.4, 2.6))

inv_N_fine = np.linspace(0, 0.15, 100)

ax.plot(inv_N_fine, np.polyval(p2, inv_N_fine), 'b-', lw=0.8, alpha=0.5)
ax.plot(inv_N, a1_poly2, 'bo', ms=5, label=r'poly2 fit')

ax.plot(inv_N_fine, np.polyval(p3, inv_N_fine), 'r--', lw=0.8, alpha=0.5)
ax.plot(inv_N, a1_poly3, 'r^', ms=5, label=r'poly3 fit')

# Mark extrapolated values
ax.plot(0, a1_inf_p2, 'bs', ms=6, mfc='none', mew=1.2)
ax.plot(0, a1_inf_p3, 'r^', ms=6, mfc='none', mew=1.2)

ax.annotate(rf'$a_1^\infty = {a1_inf_p2:.4f}$',
            xy=(0.005, a1_inf_p2), fontsize=7.5,
            xytext=(0.04, a1_inf_p2 + 0.002),
            arrowprops=dict(arrowstyle='->', color='b', lw=0.7),
            color='b')

# Axis labels
ax.set_xlabel(r'$1/N$')
ax.set_ylabel(r'$a_1$')
ax.set_xlim(-0.01, 0.14)

# Add top axis with N values
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
N_ticks = [8, 16, 32]
ax2.set_xticks([1/n for n in N_ticks])
ax2.set_xticklabels([str(n) for n in N_ticks])
ax2.set_xlabel(r'$N$ (particles in periodic box)')

ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig('figS_eos_convergence.pdf', dpi=600, bbox_inches='tight')
print(f'Saved figS_eos_convergence.pdf')
print(f'  poly2: a1_inf = {a1_inf_p2:.6f}')
print(f'  poly3: a1_inf = {a1_inf_p3:.6f}')
plt.close()
