#!/bin/bash
PARAM1=(intact paragraph word rest)
PARAM2=(100 700)
PARAM3_All=(\
  PCA IncrementalPCA SparsePCA MiniBatchSparsePCA \
  KernelPCA FastICA FactorAnalysis TruncatedSVD \
  DictionaryLearning MiniBatchDictionaryLearning \
  Isomap SpectralEmbedding \
  LocallyLinearEmbedding MDS UMAP \
  eigenvector_centrality pagerank_centrality strength \
  )

PARAM3_Filtered=(\
PCA FactorAnalysis IncrementalPCA TruncatedSVD
)
PARAM3_GraphMeasures=(\
eigenvector_centrality pagerank_centrality strength
)
KERNELS=(gaussian mexican_hat laplace)
WIDTHS=($(seq 5 5 50))
ITERATIONS=($(seq 1 1 1))

for cond in "${PARAM1[@]}"; do
  for fac in "${PARAM2[@]}"; do
    for alg in "${PARAM3_Filtered[@]}" "${PARAM3_GraphMeasures[@]}"; do
      for ker in "${KERNELS[@]}"; do
        for w in "${WIDTHS[@]}"; do
	  for i in "${ITERATIONS[@]}"; do
            # keep “10 10 isfc” constant as before
            echo "$cond $fac 10 10 isfc $alg $w $ker $i"
    	  done
        done
      done
    done
  done
done


# How to runs
# chmod +x make_combos.sh
# Rename combos.txt to whatever file name you want
# ./make_combos.sh > combos.txt
