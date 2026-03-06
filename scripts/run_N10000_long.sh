#!/bin/zsh
# N=10000 GPE dynamics with t_max=10 (longer evolution)
set -e

BASEDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASEDIR"/hydro

echo "============================================================"
echo "  GPE dynamics, N=10000, t_max=10"
echo "============================================================"

python3 run_hydro_largeN.py \
    --eos_file ../eos/eos_fit_weak.npz \
    --eos_model poly2 \
    --N_part 10000 \
    --omega_rho_quench 2.0 \
    --t_max 10.0 \
    --N_r 320 \
    --N_z 640 \
    --cfl 0.3 \
    --output results/hydro_results_N10000_long.npz

echo "Done at $(date)"
