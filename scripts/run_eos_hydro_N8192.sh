#!/bin/zsh
#
# Pipeline: improved EOS (C3=1e-3) + N=8192 GPE hydro
#
# Run on a machine with GPU (for EOS VMC) and decent CPU (for hydro).
# Assumes the repo is cloned at $REPO.
#
# Usage:
#   zsh run_eos_hydro_N8192.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repo root: $REPO"

# ─── Step 1: High-quality EOS for C3=1e-3, C6=1e-6 ───
echo ""
echo "========================================"
echo " Step 1: Uniform VMC EOS (C3=1e-3)"
echo "========================================"

cd "$REPO/eos"

python3 uniform_vmc_eos.py \
    --C3 1e-3 \
    --C6 1e-6 \
    --N_part 8 \
    --n_walkers 32768 \
    --n_equil 300 \
    --n_meas 1000 \
    --thin 5 \
    --n_rho 40 \
    --rho_min 0.05 \
    --rho_max 15.0 \
    --output eos_vmc_C3_1e-3_hq.npz \
    --seed 12345

echo ""
echo "EOS VMC done. Fitting..."

python3 fit_eos.py \
    --input eos_vmc_C3_1e-3_hq.npz \
    --output eos_fit_C3_1e-3_hq.npz

echo ""
echo "EOS fit saved to eos_fit_C3_1e-3_hq.npz"

# ─── Step 2: N=8192 GPE hydro with improved EOS ───
echo ""
echo "========================================"
echo " Step 2: GPE Hydro N=8192 (C3=1e-3)"
echo "========================================"

cd "$REPO/hydro"

# N=8192 with C3=1e-3: cloud is larger, need bigger domain.
# R_TF ~ (15*N*g_eff/(4*pi))^0.2 with g_eff ~ 0.22
# R_TF ~ (15*8192*0.22/(4*pi))^0.2 ~ (8588)^0.2 ~ 6.3
# R_max ~ 2.5*R_TF ~ 16, Z_max ~ 16
# Use 256x512 grid for adequate resolution.
# t_max=6 to capture several breathing periods.

python3 run_hydro_largeN.py \
    --eos_file "$REPO/eos/eos_fit_C3_1e-3_hq.npz" \
    --eos_model poly2 \
    --N_part 8192 \
    --omega_rho_quench 2.0 \
    --t_max 6.0 \
    --N_r 256 \
    --N_z 512 \
    --cfl 0.3 \
    --output "$REPO/hydro/results/hydro_results_N8192.npz"

echo ""
echo "========================================"
echo " Done! Results saved to:"
echo "   EOS: $REPO/eos/eos_fit_C3_1e-3_hq.npz"
echo "   Hydro: $REPO/hydro/results/hydro_results_N8192.npz"
echo "========================================"
