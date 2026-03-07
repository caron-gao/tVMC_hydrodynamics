#!/bin/zsh
#
# Finite-size convergence test for uniform VMC EOS
#
# Runs the same EOS calculation with N=8, 16, 32 particles
# to check convergence of the polynomial coefficients.
#
# Usage:
#   zsh run_eos_finite_size.sh              # Run all three (N=8, 16, 32)
#   zsh run_eos_finite_size.sh 16           # Run only N=16
#   zsh run_eos_finite_size.sh 8 16 32      # Specify which N values
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repo root: $REPO"

cd "$REPO/eos"

# Default: run N=8, 16, 32
if [[ $# -eq 0 ]]; then
    N_VALUES=(8 16 32)
else
    N_VALUES=("$@")
fi

# Common parameters (same as production run)
C3="1e-3"
C6="1e-6"
N_WALKERS=32768
N_EQUIL=300
N_MEAS=5000
THIN=5
N_RHO=20
RHO_MIN=1
RHO_MAX=200
SEED=12345

for N_PART in "${N_VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo " EOS: N_part=${N_PART}, rho=[${RHO_MIN}, ${RHO_MAX}]"
    echo "========================================"

    OUT_VMC="eos_vmc_C3_1e-3_N${N_PART}.npz"
    OUT_FIT="eos_fit_C3_1e-3_N${N_PART}.npz"

    python3 uniform_vmc_eos.py \
        --C3 $C3 \
        --C6 $C6 \
        --N_part $N_PART \
        --n_walkers $N_WALKERS \
        --n_equil $N_EQUIL \
        --n_meas $N_MEAS \
        --thin $THIN \
        --n_rho $N_RHO \
        --rho_min $RHO_MIN \
        --rho_max $RHO_MAX \
        --output $OUT_VMC \
        --seed $SEED

    echo ""
    echo "Fitting N=${N_PART}..."

    python3 fit_eos.py \
        --input $OUT_VMC \
        --output $OUT_FIT

    echo "  -> $OUT_FIT"
done

echo ""
echo "========================================"
echo " All done! Compare results with:"
echo ""
echo "   python3 -c \""
echo "   import numpy as np"
echo "   for N in [${(j:, :)N_VALUES}]:"
echo "       f = np.load(f'eos_fit_C3_1e-3_N{N}.npz', allow_pickle=True)"
echo "       p2 = f['poly2_coeffs']"
echo "       p3 = f['poly3_coeffs']"
echo "       print(f'N={N:3d}: a1={p2[0]:.6f} a2={p2[1]:.6f}  (poly3: {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.2e})')"
echo "   \""
echo "========================================"
