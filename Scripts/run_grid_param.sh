#!/bin/bash
#SBATCH --job-name=Trazz_Timecorr_Param_Sweep
#SBATCH --output=slurm-out/param-sweep.%a.out
#SBATCH --error=slurm-out/param-sweep.%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-15            # 10 tasks: IDs 0–9
#SBATCH --mail-type=all
#SBATCH --mail-user=tp183485@umconnect.umt.edu

# 1) go into your repo and activate your .venv
cd /mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp
source .venv/bin/activate

# 2) pick the weight‐function name based on the array ID
#    (make sure the order here matches the 10 you want)
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

# 3) run your script
python Cluster_Grid_Param_TRAZZ.py intact 100 10 10 isfc "$WP" 5 gaussian

echo "[$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] exit code: $?"