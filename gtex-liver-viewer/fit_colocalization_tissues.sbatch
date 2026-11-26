#!/bin/bash
#SBATCH --job-name=gtex_coloc_fit
#SBATCH --partition=shortq
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/fit_colocalization_%A_%a.out
#SBATCH --error=logs/fit_colocalization_%A_%a.out

# Fit one GTEx tissue's colocalization scale per array task: fit_colocalization.py --tissue on
# <root>/<slug>, writing <out-dir>/<slug>.json, which serve.py --tissues <root> --colocalization-dir <out-dir>
# serves. The array index is the tissue's `index` in <root>/tissues.csv (gtex-liver-cells' layout).
# Submit from this folder:
#
#   sbatch --array=0-39%10 fit_colocalization_tissues.sbatch <root> <out-dir> [<python>]
#
# <python> defaults to the chromatic-domains study's venv (README.md), whose numpy and scipy the fit needs.
# Memory: the fit holds each slide's 100 nearest neighbours as int64, 800 B a cell; GTEx's largest slide
# (spleen, 2.7M cells) peaks at about 4 GB. Time: the liver cohort's 126M cells took 6.7 min on an M4.

set -euo pipefail

ROOT="${1:?usage: sbatch --array=<indices> fit_colocalization_tissues.sbatch <root> <out-dir> [<python>]}"
OUT_DIR="${2:?usage: sbatch --array=<indices> fit_colocalization_tissues.sbatch <root> <out-dir> [<python>]}"
# The folder the job starts in, which sbatch makes the one it was submitted from (SLURM_SUBMIT_DIR can name
# another view of it, as in a container); it must hold fit_colocalization.py.
VIEWER_DIR="$PWD"
if [ ! -f "$VIEWER_DIR/fit_colocalization.py" ]; then
  echo "submit from the viewer's folder: $VIEWER_DIR has no fit_colocalization.py" >&2
  exit 1
fi
PYTHON="${3:-$VIEWER_DIR/../gtex-liver-chromatic-domains/.venv/bin/python}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array job: sbatch --array=<indices from tissues.csv>}"

SLUG="$(awk -F, -v i="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == i { print $2 }' "$ROOT/tissues.csv")"
if [ -z "$SLUG" ]; then
  echo "no tissue with index $SLURM_ARRAY_TASK_ID in $ROOT/tissues.csv" >&2
  exit 1
fi

echo "node:    $(hostname)"
echo "task:    $SLURM_ARRAY_TASK_ID ($SLUG)"
echo "root:    $ROOT"
echo "out:     $OUT_DIR"
echo "python:  $PYTHON"
echo "start:   $(date -Is)"

cd "$VIEWER_DIR"
mkdir -p "$OUT_DIR"
"$PYTHON" fit_colocalization.py --tissue "$ROOT/$SLUG" --out-dir "$OUT_DIR"
echo "end:     $(date -Is)"
