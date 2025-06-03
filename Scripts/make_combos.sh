#!/bin/bash
PARAM1=(intact paragraph word rest)
PARAM2=(100)
PARAM3=(\
  PCA IncrementalPCA SparsePCA MiniBatchSparsePCA \
  KernelPCA FastICA FactorAnalysis TruncatedSVD \
  DictionaryLearning MiniBatchDictionaryLearning \
  Isomap SpectralEmbedding \
  LocallyLinearEmbedding MDS UMAP)
KERNELS=(gaussian mexican_hat laplace)
WIDTHS=($(seq 5 5 50))

: > combos.txt
for cond in "${PARAM1[@]}"; do
  for fac in "${PARAM2[@]}"; do
    for alg in "${PARAM3[@]}"; do
      for ker in "${KERNELS[@]}"; do
        for w in "${WIDTHS[@]}"; do
          # keep “10 10 isfc” constant as before
          echo "$cond $fac 10 10 isfc $alg $w $ker"
        done
      done
    done
  done
done


# How to run
# chmod +x make_combos.sh
# ./make_combos.sh > combos.txt
