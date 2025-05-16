#!/bin/bash
#SBATCH --job-name=Trazz_Timecorr_Param_Sweep
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-14
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tp183485@umconnect.umt.edu

LOGDIR=/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/slurm-out
mkdir -p "$LOGDIR"

cd /mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp
source timecorr_venv/bin/activate
cd Scripts

PARAMS=(
  PCA
  IncrementalPCA
  SparsePCA
  MiniBatchSparsePCA
  KernelPCA
  FastICA
  FactorAnalysis
  TruncatedSVD
  DictionaryLearning
  MiniBatchDictionaryLearning
  TSNE
  Isomap
  SpectralEmbedding
  LocallyLinearEmbedding
  MDS
  UMAP
)

WP=${PARAMS[$SLURM_ARRAY_TASK_ID]}

OUT="$LOGDIR/param-sweep.${WP}.out"
ERR="$LOGDIR/param-sweep.${WP}.err"

# Redirect everything from here on
exec >"$OUT" 2>"$ERR"


python Cluster_Grid_Param_TRAZZ.py intact 100 10 10 isfc "$WP" 5 gaussian

echo "[$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] ($WP) exit code: $?" 
