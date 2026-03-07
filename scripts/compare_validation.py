#!/usr/bin/env python3
"""
Compare validation results: GPE (N=32 EOS) vs tVMC.
Also checks N=8192 grid convergence.
Run after run_validation_tests.sh completes.
"""
import numpy as np
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'hydro', 'results')
DATA = os.path.join(REPO, 'data')

def load_safe(path):
    try:
        return np.load(path, allow_pickle=True)
    except FileNotFoundError:
        return None

def breathing_freq(t, obs):
    """Extract dominant frequency from FFT."""
    dt = t[1] - t[0]
    centered = obs - obs.mean()
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    power = np.abs(np.fft.rfft(centered))**2
    omega = 2 * np.pi * freqs
    idx = np.argmax(power[1:]) + 1
    return omega[idx]

print("=" * 70)
print("  Validation comparison: GPE (N=32 EOS) vs tVMC")
print("=" * 70)

# ─── N=8 comparison ───
print("\n── N=8 ──")
gpe = load_safe(os.path.join(RESULTS, 'hydro_results_N8_N32eos.npz'))
tvmc = load_safe(os.path.join(DATA, 'tvmc_N8_freeze_u2', 'results.npz'))
gpe_old = load_safe(os.path.join(RESULTS, 'hydro_results_N8.npz'))

if gpe is not None:
    t_g = gpe['times']
    rho2_g = gpe['rho2']
    z2_g = gpe['z2']
    E_g = gpe['energy']
    E_pre = float(gpe['E_pre'])
    rho2_0 = float(gpe['rho2_0'])
    z2_0 = float(gpe['z2_0'])
    omega_g = breathing_freq(t_g, rho2_g)

    print(f"  GPE (N=32 EOS):  E_gs={E_pre:.2f}, "
          f"⟨ρ²⟩₀={rho2_0:.4f}, ⟨z²⟩₀={z2_0:.4f}")
    print(f"    ω_br={omega_g:.3f}, "
          f"⟨ρ²⟩ range=[{rho2_g.min():.3f}, {rho2_g.max():.3f}], "
          f"⟨z²⟩ range=[{z2_g.min():.3f}, {z2_g.max():.3f}]")
    print(f"    δE/E={abs(E_g.max()-E_g.min())/abs(E_g.mean()):.2e}, "
          f"δN/N={abs(gpe['particle_number'].max()-gpe['particle_number'].min())/gpe['particle_number'].mean():.2e}")

    if tvmc is not None:
        t_t = tvmc['times']
        rho2_t = tvmc['rho2_values']
        z2_t = tvmc['z2_values']
        E_t = tvmc['energies'] if 'energies' in tvmc.files else None

        # tVMC ground state (t=0 values)
        print(f"  tVMC (frozen-u₂): "
              f"⟨ρ²⟩₀={rho2_t[0]:.4f}, ⟨z²⟩₀={z2_t[0]:.4f}")

        # Differences
        dr2 = abs(rho2_0 - rho2_t[0]) / rho2_t[0] * 100
        dz2 = abs(z2_0 - z2_t[0]) / z2_t[0] * 100
        print(f"  Δ⟨ρ²⟩₀/⟨ρ²⟩₀ = {dr2:.2f}%")
        print(f"  Δ⟨z²⟩₀/⟨z²⟩₀ = {dz2:.2f}%")

        omega_t = breathing_freq(t_t, rho2_t)
        print(f"  ω_br: GPE={omega_g:.3f}, tVMC={omega_t:.3f}, "
              f"Δω/ω={abs(omega_g-omega_t)/omega_t*100:.2f}%")
else:
    print("  [Not found: hydro_results_N8_N32eos.npz]")

# ─── N=128 comparison ───
print("\n── N=128 ──")
gpe = load_safe(os.path.join(RESULTS, 'hydro_results_N128_N32eos.npz'))
tvmc = load_safe(os.path.join(DATA, 'tvmc_N128_freeze_u2', 'results.npz'))
gpe_old = load_safe(os.path.join(RESULTS, 'hydro_results_N128_new_gs_poly2.npz'))

if gpe is not None:
    t_g = gpe['times']
    rho2_g = gpe['rho2']
    z2_g = gpe['z2']
    E_g = gpe['energy']
    E_pre = float(gpe['E_pre'])
    rho2_0 = float(gpe['rho2_0'])
    z2_0 = float(gpe['z2_0'])
    omega_g = breathing_freq(t_g, rho2_g)

    print(f"  GPE (N=32 EOS):  E_gs={E_pre:.2f}, "
          f"⟨ρ²⟩₀={rho2_0:.4f}, ⟨z²⟩₀={z2_0:.4f}")
    print(f"    ω_br={omega_g:.3f}, "
          f"⟨ρ²⟩ range=[{rho2_g.min():.3f}, {rho2_g.max():.3f}], "
          f"⟨z²⟩ range=[{z2_g.min():.3f}, {z2_g.max():.3f}]")
    print(f"    δE/E={abs(E_g.max()-E_g.min())/abs(E_g.mean()):.2e}")

    if tvmc is not None:
        t_t = tvmc['times']
        rho2_t = tvmc['rho2_values']
        z2_t = tvmc['z2_values']

        print(f"  tVMC (frozen-u₂): "
              f"⟨ρ²⟩₀={rho2_t[0]:.4f}, ⟨z²⟩₀={z2_t[0]:.4f}")

        dr2 = abs(rho2_0 - rho2_t[0]) / rho2_t[0] * 100
        dz2 = abs(z2_0 - z2_t[0]) / z2_t[0] * 100
        print(f"  Δ⟨ρ²⟩₀/⟨ρ²⟩₀ = {dr2:.2f}%")
        print(f"  Δ⟨z²⟩₀/⟨z²⟩₀ = {dz2:.2f}%")

        omega_t = breathing_freq(t_t, rho2_t)
        print(f"  ω_br: GPE={omega_g:.3f}, tVMC={omega_t:.3f}, "
              f"Δω/ω={abs(omega_g-omega_t)/omega_t*100:.2f}%")

    if gpe_old is not None:
        E_old = float(gpe_old['E_pre'])
        print(f"\n  Comparison with OLD EOS (a₁=0.111):")
        print(f"    E_gs: old={E_old:.2f}, new={E_pre:.2f}")
        print(f"    ⟨ρ²⟩₀: old={float(gpe_old['rho2_0']):.4f}, "
              f"new={rho2_0:.4f}")
else:
    print("  [Not found: hydro_results_N128_N32eos.npz]")

# ─── N=8192 grid convergence ───
print("\n── N=8192 grid convergence ──")
gpe_std = load_safe(os.path.join(RESULTS, 'hydro_results_N8192.npz'))
gpe_fine = load_safe(os.path.join(RESULTS, 'hydro_results_N8192_fine.npz'))

if gpe_std is not None and gpe_fine is not None:
    for label, d in [("256×512 (standard)", gpe_std),
                     ("320×640 (fine)", gpe_fine)]:
        t = d['times']
        rho2 = d['rho2']
        z2 = d['z2']
        E = d['energy']
        omega = breathing_freq(t, rho2)
        dE = abs(E.max() - E.min()) / abs(E.mean())
        print(f"  {label}:")
        print(f"    ω_br={omega:.3f}, "
              f"⟨ρ²⟩=[{rho2.min():.4f}, {rho2.max():.4f}], "
              f"⟨z²⟩=[{z2.min():.4f}, {z2.max():.4f}]")
        print(f"    E_gs={float(d['E_pre']):.2f}, δE/E={dE:.2e}")

    # Quantify differences
    omega_s = breathing_freq(gpe_std['times'], gpe_std['rho2'])
    omega_f = breathing_freq(gpe_fine['times'], gpe_fine['rho2'])
    print(f"\n  Grid convergence:")
    print(f"    Δω/ω = {abs(omega_s-omega_f)/omega_s*100:.3f}%")
    print(f"    ΔE_gs/E_gs = "
          f"{abs(float(gpe_std['E_pre'])-float(gpe_fine['E_pre']))/abs(float(gpe_std['E_pre']))*100:.4f}%")
    print(f"    Δ⟨ρ²⟩₀ = "
          f"{abs(float(gpe_std['rho2_0'])-float(gpe_fine['rho2_0'])):.6f}")
elif gpe_std is not None:
    print("  [Standard grid found, fine grid not yet available]")
    print("  Run: zsh scripts/run_validation_tests.sh --N8192")
elif gpe_fine is not None:
    print("  [Fine grid found, standard grid not found]")
else:
    print("  [No N=8192 results found]")

# ─── Finite-size EOS convergence ───
print("\n── Finite-size EOS convergence ──")
print(f"  {'N':>4s}  {'a₁ (poly2)':>12s}  {'a₂ (poly2)':>12s}")
a1_vals = []
N_vals = [8, 16, 32]
for N in N_vals:
    f = os.path.join(REPO, 'eos', f'eos_fit_C3_1e-3_N{N}.npz')
    d = load_safe(f)
    if d is not None:
        p2 = d['poly2_coeffs']
        print(f"  {N:4d}  {p2[0]:12.6f}  {p2[1]:12.2e}")
        a1_vals.append(p2[0])
    else:
        print(f"  {N:4d}  [not found]")

if len(a1_vals) == 3:
    a1_vals = np.array(a1_vals)
    inv_N = 1.0 / np.array(N_vals)
    p = np.polyfit(inv_N, a1_vals, 1)
    a1_inf = p[1]
    alpha = -p[0] / a1_inf
    rich = (32*a1_vals[2] - 16*a1_vals[1]) / 16
    print(f"\n  Linear extrapolation: a₁∞ = {a1_inf:.6f}, α = {alpha:.3f}")
    print(f"  Richardson (16,32):   a₁∞ = {rich:.6f}")
    print(f"  a₁(N=32) / a₁∞ = {a1_vals[2]/a1_inf:.4f} "
          f"({(1-a1_vals[2]/a1_inf)*100:.1f}% deviation)")

print("\n" + "=" * 70)
print("  Done.")
print("=" * 70)
