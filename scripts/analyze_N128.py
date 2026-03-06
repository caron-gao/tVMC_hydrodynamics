#!/usr/bin/env python3
"""
Analysis script comparing tVMC, GPE (ground-state IC), and GPE (Gaussian IC)
for N=128 breathing dynamics.

Produces:
  - hydro_comparison_N128.png (multi-panel figure)
  - Printed summary table
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# File paths
# ──────────────────────────────────────────────────────────────────────────────
TVMC_FILE   = "tvmc_output_N128/results.npz"
HYDRO_GS    = "hydro_results_N128_gs.npz"
HYDRO_GAUSS = "hydro_results_N128_gauss.npz"
OUT_FIG     = "hydro_comparison_N128.png"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: FFT frequency analysis
# ──────────────────────────────────────────────────────────────────────────────
def fft_dominant_freq(times, signal):
    """Return (freqs, power, dominant_angular_freq) from a real signal.

    The signal is detrended (mean-subtracted) before FFT.
    Returns angular frequency omega = 2*pi*f.
    """
    N = len(signal)
    dt_mean = (times[-1] - times[0]) / (N - 1)

    # Uniform resampling (in case times are not perfectly uniform)
    t_uniform = np.linspace(times[0], times[-1], N)
    sig_uniform = np.interp(t_uniform, times, signal)

    sig_detrend = sig_uniform - sig_uniform.mean()

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(N)
    sig_windowed = sig_detrend * window

    fft_vals = np.fft.rfft(sig_windowed)
    freqs = np.fft.rfftfreq(N, d=dt_mean)
    power = np.abs(fft_vals)**2

    # Skip DC component (index 0)
    idx_peak = np.argmax(power[1:]) + 1
    f_peak = freqs[idx_peak]
    omega_peak = 2 * np.pi * f_peak

    return freqs, power, omega_peak


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
print("Loading data...")

tvmc = np.load(TVMC_FILE, allow_pickle=True)
hgs  = np.load(HYDRO_GS, allow_pickle=True)
hgau = np.load(HYDRO_GAUSS, allow_pickle=True)

# tVMC
tvmc_times = tvmc['times']
tvmc_rho2  = tvmc['rho2_values']
tvmc_z2    = tvmc['z2_values']
tvmc_E_gs  = float(tvmc['E_gs'])
tvmc_rho2_gs = float(tvmc['rho2_gs'])
tvmc_z2_gs   = float(tvmc['z2_gs'])

# GPE ground-state IC
gs_times = hgs['times']
gs_rho2  = hgs['rho2']
gs_z2    = hgs['z2']
gs_energy = hgs['energy']
gs_N      = hgs['particle_number']
gs_E_post = float(hgs['E_post'])
gs_E_pre  = float(hgs['E_pre'])
gs_rho2_0 = float(hgs['rho2_0'])
gs_z2_0   = float(hgs['z2_0'])
# Scaling ansatz
gs_sc_times = hgs['scaling_times']
gs_sc_rho2  = hgs['scaling_rho2']
gs_sc_z2    = hgs['scaling_z2']

# GPE Gaussian IC
gau_times = hgau['times']
gau_rho2  = hgau['rho2']
gau_z2    = hgau['z2']
gau_energy = hgau['energy']
gau_N      = hgau['particle_number']
gau_E_post = float(hgau['E_post'])
gau_E_pre  = float(hgau['E_pre'])
gau_rho2_0 = float(hgau['rho2_0'])
gau_z2_0   = float(hgau['z2_0'])
gau_sc_times = hgau['scaling_times']
gau_sc_rho2  = hgau['scaling_rho2']
gau_sc_z2    = hgau['scaling_z2']

print("  tVMC:      {} time points, t in [{:.4f}, {:.4f}]".format(
    len(tvmc_times), tvmc_times[0], tvmc_times[-1]))
print("  GPE-GS:    {} time points, t in [{:.4f}, {:.4f}]".format(
    len(gs_times), gs_times[0], gs_times[-1]))
print("  GPE-Gauss: {} time points, t in [{:.4f}, {:.4f}]".format(
    len(gau_times), gau_times[0], gau_times[-1]))


# ──────────────────────────────────────────────────────────────────────────────
# FFT analysis
# ──────────────────────────────────────────────────────────────────────────────
print("\nFFT frequency analysis on rho2(t)...")

# Use a transient-free window: skip early times to let oscillation settle
# For tVMC, use full signal (it starts from quench)
tvmc_freqs, tvmc_power, tvmc_omega = fft_dominant_freq(tvmc_times, tvmc_rho2)
gs_freqs, gs_power, gs_omega       = fft_dominant_freq(gs_times, gs_rho2)
gau_freqs, gau_power, gau_omega    = fft_dominant_freq(gau_times, gau_rho2)

print(f"  tVMC      dominant omega_rho2 = {tvmc_omega:.4f}")
print(f"  GPE-GS    dominant omega_rho2 = {gs_omega:.4f}")
print(f"  GPE-Gauss dominant omega_rho2 = {gau_omega:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating figure...")

fig, axes = plt.subplots(3, 1, figsize=(11, 13))

# Colors
c_tvmc  = '#d62728'   # red
c_gs    = '#1f77b4'   # blue
c_gauss = '#2ca02c'   # green
c_sc    = '#9467bd'   # purple (scaling ansatz)

# ── Panel 1: rho2(t) ────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(tvmc_times, tvmc_rho2, color=c_tvmc, lw=1.2, alpha=0.85,
        label='tVMC')
ax.plot(gs_times, gs_rho2, color=c_gs, lw=1.8,
        label='GPE (GS IC)')
ax.plot(gau_times, gau_rho2, color=c_gauss, lw=1.8, ls='--',
        label='GPE (Gauss IC)')
ax.plot(gs_sc_times, gs_sc_rho2, color=c_sc, lw=1.2, ls=':',
        label='Scaling ansatz (GS)')
ax.set_ylabel(r'$\langle \rho^2 \rangle$', fontsize=13)
ax.set_xlabel('Time', fontsize=12)
ax.set_title(r'N=128 breathing dynamics:  $\omega_\rho\!: 1 \to 2$',
             fontsize=14)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(tvmc_times[-1], gs_times[-1]))

# ── Panel 2: z2(t) ──────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(tvmc_times, tvmc_z2, color=c_tvmc, lw=1.2, alpha=0.85,
        label='tVMC')
ax.plot(gs_times, gs_z2, color=c_gs, lw=1.8,
        label='GPE (GS IC)')
ax.plot(gau_times, gau_z2, color=c_gauss, lw=1.8, ls='--',
        label='GPE (Gauss IC)')
ax.plot(gs_sc_times, gs_sc_z2, color=c_sc, lw=1.2, ls=':',
        label='Scaling ansatz (GS)')
ax.set_ylabel(r'$\langle z^2 \rangle$', fontsize=13)
ax.set_xlabel('Time', fontsize=12)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(tvmc_times[-1], gs_times[-1]))

# ── Panel 3: FFT power spectra ──────────────────────────────────────────────
ax = axes[2]
# Convert to angular frequency for x-axis
tvmc_omega_axis = 2 * np.pi * tvmc_freqs
gs_omega_axis   = 2 * np.pi * gs_freqs
gau_omega_axis  = 2 * np.pi * gau_freqs

# Normalize power spectra to peak=1 for comparison
tvmc_pnorm = tvmc_power / tvmc_power[1:].max()
gs_pnorm   = gs_power / gs_power[1:].max()
gau_pnorm  = gau_power / gau_power[1:].max()

ax.plot(tvmc_omega_axis, tvmc_pnorm, color=c_tvmc, lw=1.2, alpha=0.85,
        label=f'tVMC ($\\omega$={tvmc_omega:.2f})')
ax.plot(gs_omega_axis, gs_pnorm, color=c_gs, lw=1.8,
        label=f'GPE-GS ($\\omega$={gs_omega:.2f})')
ax.plot(gau_omega_axis, gau_pnorm, color=c_gauss, lw=1.8, ls='--',
        label=f'GPE-Gauss ($\\omega$={gau_omega:.2f})')

ax.set_xlabel(r'$\omega$ (angular frequency)', fontsize=12)
ax.set_ylabel('Normalized FFT power', fontsize=13)
ax.set_title(r'FFT of $\langle \rho^2 \rangle(t)$', fontsize=14)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 25)  # Focus on low frequencies
ax.set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
print(f"  Figure saved to {OUT_FIG}")
plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
def conservation_quality(energy, N_arr, E0, N0):
    """Return (max |dN/N|, max |dE/E|)."""
    dN = np.max(np.abs(N_arr - N0) / N0)
    dE = np.max(np.abs(energy - E0) / max(abs(E0), 1e-10))
    return dN, dE


gs_dN,  gs_dE  = conservation_quality(gs_energy, gs_N, gs_E_post, 128.0)
gau_dN, gau_dE = conservation_quality(gau_energy, gau_N, gau_E_post, 128.0)

sep = "-" * 80
print("\n" + "=" * 80)
print("SUMMARY TABLE: N=128 trap quench (omega_rho: 1 -> 2)")
print("=" * 80)

# Ground state properties
print(f"\n{'Property':<30s}  {'tVMC':>14s}  {'GPE-GS':>14s}  {'GPE-Gauss':>14s}")
print(sep)
print(f"{'E_gs (pre-quench)':<30s}  {tvmc_E_gs:>14.4f}  {gs_E_pre:>14.4f}  {gau_E_pre:>14.4f}")
print(f"{'E (post-quench)':<30s}  {'---':>14s}  {gs_E_post:>14.4f}  {gau_E_post:>14.4f}")
print(f"{'<rho2>_gs':<30s}  {tvmc_rho2_gs:>14.4f}  {gs_rho2_0:>14.4f}  {gau_rho2_0:>14.4f}")
print(f"{'<z2>_gs':<30s}  {tvmc_z2_gs:>14.4f}  {gs_z2_0:>14.4f}  {gau_z2_0:>14.4f}")

# Breathing frequency
print(f"\n{'Breathing freq omega':<30s}  {tvmc_omega:>14.4f}  {gs_omega:>14.4f}  {gau_omega:>14.4f}")

# Amplitude ranges
print(f"\n{'rho2 min':<30s}  {tvmc_rho2.min():>14.4f}  {gs_rho2.min():>14.4f}  {gau_rho2.min():>14.4f}")
print(f"{'rho2 max':<30s}  {tvmc_rho2.max():>14.4f}  {gs_rho2.max():>14.4f}  {gau_rho2.max():>14.4f}")
print(f"{'rho2 amplitude':<30s}  {tvmc_rho2.max()-tvmc_rho2.min():>14.4f}  {gs_rho2.max()-gs_rho2.min():>14.4f}  {gau_rho2.max()-gau_rho2.min():>14.4f}")

print(f"\n{'z2 min':<30s}  {tvmc_z2.min():>14.4f}  {gs_z2.min():>14.4f}  {gau_z2.min():>14.4f}")
print(f"{'z2 max':<30s}  {tvmc_z2.max():>14.4f}  {gs_z2.max():>14.4f}  {gau_z2.max():>14.4f}")
print(f"{'z2 amplitude':<30s}  {tvmc_z2.max()-tvmc_z2.min():>14.4f}  {gs_z2.max()-gs_z2.min():>14.4f}  {gau_z2.max()-gau_z2.min():>14.4f}")

# Conservation
print(f"\n{'max |dN/N|':<30s}  {'---':>14s}  {gs_dN:>14.2e}  {gau_dN:>14.2e}")
print(f"{'max |dE/E|':<30s}  {'---':>14s}  {gs_dE:>14.2e}  {gau_dE:>14.2e}")

# Relative errors vs tVMC
print(f"\n{'--- Relative errors vs tVMC ---'}")
print(f"{'E_gs rel error':<30s}  {'':>14s}  {abs(gs_E_pre - tvmc_E_gs)/abs(tvmc_E_gs):>14.4%}  {abs(gau_E_pre - tvmc_E_gs)/abs(tvmc_E_gs):>14.4%}")
print(f"{'<rho2>_gs rel error':<30s}  {'':>14s}  {abs(gs_rho2_0 - tvmc_rho2_gs)/tvmc_rho2_gs:>14.4%}  {abs(gau_rho2_0 - tvmc_rho2_gs)/tvmc_rho2_gs:>14.4%}")
print(f"{'<z2>_gs rel error':<30s}  {'':>14s}  {abs(gs_z2_0 - tvmc_z2_gs)/tvmc_z2_gs:>14.4%}  {abs(gau_z2_0 - tvmc_z2_gs)/tvmc_z2_gs:>14.4%}")
print(f"{'omega rel error':<30s}  {'':>14s}  {abs(gs_omega - tvmc_omega)/tvmc_omega:>14.4%}  {abs(gau_omega - tvmc_omega)/tvmc_omega:>14.4%}")

print("\n" + "=" * 80)
print("Done.")
