#!/bin/bash
# submit_grid.sh
#SBATCH --job-name=Trazz_Helper_Script
#SBATCH --output=submit_grid.%j.out     # we’ll redirect in‐script
#SBATCH --error=sumbit_grid.%j.err
#SBATCH --time=500:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1

USER=$(whoami)
MAXJ=15
COMBOS=combos.txt
TOTAL=$(wc -l < "$COMBOS")
i=0

while [ "$i" -lt "$TOTAL" ]; do
  # count your grid jobs (RUNNING or PENDING)
  RUNNING=$(squeue -u "$USER" -n Trazz_Grid -h | wc -l)

  if [ "$RUNNING" -lt "$MAXJ" ]; then
    # grab the next line (1-based)
    LINE=$(sed -n "$((i+1))p" "$COMBOS")
    read cond fac lvl reps cfun alg width ker <<< "$LINE"

    sbatch --export=ALL,cond="$cond",fac="$fac",cfun="$cfun",alg="$alg",width="$width",ker="$ker" run_one.sh
    echo "Submitted #$((i+1)) → $cond $fac $alg $width $ker"
    i=$((i+1))
  else
    # wait for some jobs to finish
    sleep 30
  fi
done

echo "All $TOTAL combos submitted."