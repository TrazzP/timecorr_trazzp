#!/bin/bash
#SBATCH --job-name=Trazz_Grid
#SBATCH --output=/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/logs/%x-%j.out
#SBATCH --error=/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

# these are passed in by the submit script
# cond   ← intact|paragraph|sentence|word
# fac    ← 100|700
# cfun   ← isfc (hard‐coded)
# alg    ← one of your 16 WP methods
# width  ← 5,10,…,50
# ker    ← gaussian|mexican_hat|laplace

# build your output path
#OUTDIR=../Cluster_Data/"$cond"/"$fac"/"$ker"/"$alg"/"$width"
#mkdir -p "$OUTDIR"


cd /mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp
source timecorr_venv/bin/activate
cd Scripts

python Cluster_Grid_Param_TRAZZ.py \
  "$cond" "$fac" 10 10 "$cfun" "$alg" "$width" "$wp"
