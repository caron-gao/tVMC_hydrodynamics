#!/bin/zsh
# =============================================================================
# Large-N GPE Pipeline: EOS via VMC + GPE dynamics for N=1000 and N=10000
#
# Interaction: C3=1e-4, C6=1e-7 (10x weaker than original C3=1e-3, C6=1e-6)
# Cusp parameters tuned for this weaker interaction: kappa=0.1, eps=0.005, rc=0.01
#
# Usage:
#   zsh run_largeN_pipeline.sh           # Full pipeline
#   zsh run_largeN_pipeline.sh --eos     # EOS only (GPU needed)
#   zsh run_largeN_pipeline.sh --gpe     # GPE only (CPU, needs EOS done)
# =============================================================================

set -e

BASEDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASEDIR"

# ── Configuration ────────────────────────────────────────────────────────────
C3=1e-4
C6=1e-7
N_PART_VMC=8

# Cusp and u2 basis parameters (tuned for C3=1e-4 interaction)
KAPPA_CUSP=0.1
EPS_CUSP=0.005
RC_CUSP=0.01
ELL2_MIN=0.005
ELL2_MAX=0.3

RHO_MAX=50.0
N_RHO=30

EOS_OUTPUT="eos/eos_vmc_weak.npz"
EOS_FIT="eos/eos_fit_weak.npz"

# GPE parameters
N_LARGE_1=1000
N_LARGE_2=10000
OMEGA_QUENCH=2.0
T_MAX=3.5
CFL=0.3

# Parse arguments
RUN_EOS=true
RUN_GPE=true
if [[ "$1" == "--eos" ]]; then
    RUN_GPE=false
elif [[ "$1" == "--gpe" ]]; then
    RUN_EOS=false
fi

echo "============================================================"
echo "  Large-N GPE Pipeline"
echo "  C3=$C3, C6=$C6"
echo "  Cusp: kappa=$KAPPA_CUSP, eps=$EPS_CUSP, rc=$RC_CUSP"
echo "  u2 basis: ell2=[$ELL2_MIN, $ELL2_MAX]"
echo "  EOS: $RUN_EOS, GPE: $RUN_GPE"
echo "  Working directory: $BASEDIR"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Compute EOS via uniform VMC (needs GPU/JAX)
# =============================================================================
if $RUN_EOS; then
    echo "============================================================"
    echo "  Step 1: Uniform VMC EOS (C3=$C3, C6=$C6)"
    echo "============================================================"
    echo ""

    python3 eos/uniform_vmc_eos.py \
        --C3 $C3 \
        --C6 $C6 \
        --N_part $N_PART_VMC \
        --kappa_cusp $KAPPA_CUSP \
        --eps_cusp $EPS_CUSP \
        --rc_cusp $RC_CUSP \
        --ell2_min $ELL2_MIN \
        --ell2_max $ELL2_MAX \
        --n_walkers 32768 \
        --n_equil 500 \
        --n_meas 5000 \
        --thin 10 \
        --n_rho $N_RHO \
        --rho_min 0.05 \
        --rho_max $RHO_MAX \
        --output $EOS_OUTPUT

    echo ""
    echo "  VMC EOS saved to $EOS_OUTPUT"

    # ── Step 2: Fit EOS ──────────────────────────────────────────────────────
    echo ""
    echo "============================================================"
    echo "  Step 2: Fit EOS polynomial"
    echo "============================================================"
    echo ""

    python3 eos/fit_eos.py \
        --input $EOS_OUTPUT \
        --output $EOS_FIT \
        --plot_prefix eos/eos_weak_

    echo ""
    echo "  EOS fit saved to $EOS_FIT"
fi

# =============================================================================
# Step 3: GPE dynamics for large N (CPU-only, grid-based PDE)
# =============================================================================
if $RUN_GPE; then
    if [[ ! -f "$EOS_FIT" ]]; then
        echo "ERROR: EOS fit file $EOS_FIT not found. Run with --eos first."
        exit 1
    fi

    cd hydro

    # ── N=1000 ───────────────────────────────────────────────────────────────
    echo ""
    echo "============================================================"
    echo "  Step 3a: GPE dynamics, N=$N_LARGE_1"
    echo "============================================================"
    echo ""

    python3 run_hydro_largeN.py \
        --eos_file ../$EOS_FIT \
        --eos_model poly2 \
        --N_part $N_LARGE_1 \
        --omega_rho_quench $OMEGA_QUENCH \
        --t_max $T_MAX \
        --N_r 256 \
        --N_z 512 \
        --cfl $CFL \
        --output results/hydro_results_N${N_LARGE_1}.npz

    echo ""
    echo "  N=$N_LARGE_1 done."
    echo ""

    # ── N=10000 ──────────────────────────────────────────────────────────────
    echo "============================================================"
    echo "  Step 3b: GPE dynamics, N=$N_LARGE_2"
    echo "============================================================"
    echo ""

    python3 run_hydro_largeN.py \
        --eos_file ../$EOS_FIT \
        --eos_model poly2 \
        --N_part $N_LARGE_2 \
        --omega_rho_quench $OMEGA_QUENCH \
        --t_max $T_MAX \
        --N_r 320 \
        --N_z 640 \
        --cfl $CFL \
        --output results/hydro_results_N${N_LARGE_2}.npz

    echo ""
    echo "  N=$N_LARGE_2 done."

    cd "$BASEDIR"
fi

echo ""
echo "============================================================"
echo "  Pipeline complete at $(date)"
echo "============================================================"
