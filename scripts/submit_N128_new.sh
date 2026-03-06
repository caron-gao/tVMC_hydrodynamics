#!/bin/bash
#SBATCH --job-name=hydro_N128_new
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=hydro_N128_new_%j.log
#SBATCH --error=hydro_N128_new_%j.log

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo ""

cd /home/gyqyan/xinyuan/Documents/tVMC_reduced_theory_2
source .venv/bin/activate

# ── Run 1: GPE ground state IC, poly2 EOS ─────────────────────────────
echo "=========================================="
echo "Run 1: GPE ground state IC (poly2 EOS)"
echo "=========================================="
python3 run_hydro_N128.py \
    --ic gs \
    --eos_model poly2 \
    --N_part 128 \
    --omega_rho_quench 2.0 \
    --t_max 6.0 \
    --N_r 192 \
    --N_z 512 \
    --R_max 6.0 \
    --Z_max 8.0 \
    --cfl 0.3 \
    --eos_file eos_fit_extended.npz \
    --tvmc_file tvmc_output_N128_new/results.npz \
    --output hydro_results_N128_new_gs_poly2.npz

echo ""

# ── Run 2: GPE ground state IC, poly3 EOS ─────────────────────────────
echo "=========================================="
echo "Run 2: GPE ground state IC (poly3 EOS)"
echo "=========================================="
python3 run_hydro_N128.py \
    --ic gs \
    --eos_model poly3 \
    --N_part 128 \
    --omega_rho_quench 2.0 \
    --t_max 6.0 \
    --N_r 192 \
    --N_z 512 \
    --R_max 6.0 \
    --Z_max 8.0 \
    --cfl 0.3 \
    --eos_file eos_fit_extended.npz \
    --tvmc_file tvmc_output_N128_new/results.npz \
    --output hydro_results_N128_new_gs_poly3.npz

echo ""

# ── Run 3: Gaussian IC matched to new tVMC, poly2 EOS ─────────────────
echo "=========================================="
echo "Run 3: Gaussian IC (poly2 EOS)"
echo "=========================================="
python3 run_hydro_N128.py \
    --ic gaussian \
    --eos_model poly2 \
    --N_part 128 \
    --omega_rho_quench 2.0 \
    --t_max 6.0 \
    --N_r 192 \
    --N_z 512 \
    --R_max 6.0 \
    --Z_max 8.0 \
    --cfl 0.3 \
    --eos_file eos_fit_extended.npz \
    --tvmc_file tvmc_output_N128_new/results.npz \
    --output hydro_results_N128_new_gauss_poly2.npz

echo ""
echo "=== Job finished at $(date) ==="
