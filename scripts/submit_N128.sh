#!/bin/bash
#SBATCH --job-name=hydro_N128
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=hydro_N128_%j.log
#SBATCH --error=hydro_N128_%j.log

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo ""

cd /home/gyqyan/xinyuan/Documents/tVMC_reduced_theory_2

# ── Run 1: GPE ground state IC ──────────────────────────────────────
echo "=========================================="
echo "Run 1: GPE ground state IC"
echo "=========================================="
python3 run_hydro_N128.py \
    --ic gs \
    --N_part 128 \
    --omega_rho_quench 2.0 \
    --t_max 3.5 \
    --N_r 192 \
    --N_z 512 \
    --R_max 6.0 \
    --Z_max 8.0 \
    --cfl 0.3 \
    --eos_file eos_fit_extended.npz \
    --output hydro_results_N128_gs.npz

echo ""

# ── Run 2: Gaussian IC matched to tVMC moments ─────────────────────
echo "=========================================="
echo "Run 2: Gaussian IC matched to tVMC moments"
echo "=========================================="
python3 run_hydro_N128.py \
    --ic gaussian \
    --N_part 128 \
    --omega_rho_quench 2.0 \
    --t_max 3.5 \
    --N_r 192 \
    --N_z 512 \
    --R_max 6.0 \
    --Z_max 8.0 \
    --cfl 0.3 \
    --eos_file eos_fit_extended.npz \
    --output hydro_results_N128_gauss.npz

echo ""
echo "=== Job finished at $(date) ==="
