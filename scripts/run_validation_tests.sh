#!/bin/zsh
#
# Validation & convergence tests for SM
#
# Runs:
#   1. N=8   GPE with N=32 EOS (re-validate with converged EOS)
#   2. N=128 GPE with N=32 EOS (re-validate with converged EOS)
#   3. N=8192 grid convergence  (320x640 vs current 256x512)
#
# Usage:
#   zsh run_validation_tests.sh              # Run all three
#   zsh run_validation_tests.sh --N8         # N=8 only
#   zsh run_validation_tests.sh --N128       # N=128 only
#   zsh run_validation_tests.sh --N8192      # N=8192 grid convergence only
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
EOS_FILE="$REPO/eos/eos_fit_C3_1e-3_N32.npz"
HYDRO="$REPO/hydro/run_hydro_largeN.py"
OUTDIR="$REPO/hydro/results"

echo "Repo: $REPO"
echo "EOS:  $EOS_FILE"
echo ""

RUN_N8=true
RUN_N128=true
RUN_N8192=true

if [[ "${1:-}" == "--N8" ]]; then
    RUN_N128=false; RUN_N8192=false
elif [[ "${1:-}" == "--N128" ]]; then
    RUN_N8=false; RUN_N8192=false
elif [[ "${1:-}" == "--N8192" ]]; then
    RUN_N8=false; RUN_N128=false
fi

# ─── Test 1: N=8 GPE with N=32 EOS ───
if $RUN_N8; then
echo "========================================"
echo " Test 1: N=8 GPE with N=32 EOS"
echo "========================================"
echo "  Grid: 128x256, domain R=5 Z=5, t_max=3"

cd "$REPO/hydro"
python3 "$HYDRO" \
    --eos_file "$EOS_FILE" \
    --eos_model poly2 \
    --N_part 8 \
    --omega_rho_quench 2.0 \
    --t_max 3.0 \
    --N_r 128 \
    --N_z 256 \
    --R_max 5.0 \
    --Z_max 5.0 \
    --cfl 0.3 \
    --output "$OUTDIR/hydro_results_N8_N32eos.npz" \
    --no_plot

echo "  -> Saved: $OUTDIR/hydro_results_N8_N32eos.npz"
echo ""
fi

# ─── Test 2: N=128 GPE with N=32 EOS ───
if $RUN_N128; then
echo "========================================"
echo " Test 2: N=128 GPE with N=32 EOS"
echo "========================================"
echo "  Grid: 192x512, domain R=6 Z=8, t_max=6"

cd "$REPO/hydro"
python3 "$HYDRO" \
    --eos_file "$EOS_FILE" \
    --eos_model poly2 \
    --N_part 128 \
    --omega_rho_quench 2.0 \
    --t_max 6.0 \
    --N_r 192 \
    --N_z 512 \
    --R_max 6.0 \
    --Z_max 8.0 \
    --cfl 0.3 \
    --output "$OUTDIR/hydro_results_N128_N32eos.npz" \
    --no_plot

echo "  -> Saved: $OUTDIR/hydro_results_N128_N32eos.npz"
echo ""
fi

# ─── Test 3: N=8192 grid convergence (finer grid) ───
if $RUN_N8192; then
echo "========================================"
echo " Test 3: N=8192 grid convergence (320x640)"
echo "========================================"
echo "  Grid: 320x640 (vs current 256x512), t_max=6"

cd "$REPO/hydro"
python3 "$HYDRO" \
    --eos_file "$EOS_FILE" \
    --eos_model poly2 \
    --N_part 8192 \
    --omega_rho_quench 2.0 \
    --t_max 6.0 \
    --N_r 320 \
    --N_z 640 \
    --cfl 0.3 \
    --output "$OUTDIR/hydro_results_N8192_fine.npz" \
    --no_plot

echo "  -> Saved: $OUTDIR/hydro_results_N8192_fine.npz"
echo ""
fi

echo "========================================"
echo " All requested tests complete."
echo "========================================"
echo ""
echo "To compare results, run:"
echo "  python3 scripts/compare_validation.py"
