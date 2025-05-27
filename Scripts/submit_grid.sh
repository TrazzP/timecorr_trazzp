#!/bin/bash
#SBATCH --job-name=Trazz_Helper_Script
#SBATCH --output=submit_grid.%j.out
#SBATCH --error=submit_grid.%j.err
#SBATCH --time=500:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1

USER=$(whoami)
MAXJ=15
COMBOS=combos.txt
TOTAL=$(wc -l < "$COMBOS")
i=1  # 1-based

while [ "$i" -le "$TOTAL" ]; do
  # throttle by counting your worker‐jobs named "Trazz_Grid"
  RUNNING=$(squeue -u "$USER" -n Trazz_Grid -h | wc -l)

  if [ "$RUNNING" -lt "$MAXJ" ]; then
    # grab line i
    LINE=$(sed -n "${i}p" "$COMBOS")
    read cond fac lvl reps cfun rfun width wp <<< "$LINE"

    # build the exact output path
    OUT="/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/${cond}_${fac}_${lvl}_${reps}_${cfun}_${rfun}_${width}_${wp}.csv"

    if [ -f "$OUT" ]; then
      echo "Skipping #$i → ${cond},${fac},${lvl},${reps},${cfun},${rfun},${width},${wp} (already done)"
      ((i++))
      continue
    fi

    sbatch \
      --job-name="${cond}_${fac}_${lvl}_${reps}_${cfun}_${rfun}_${width}_${wp}" \
      --output="/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/logs/%x-%j.out" \
      --error="/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/logs/%x-%j.err" \
      --export=ALL,cond="$cond",fac="$fac",lvl="$lvl",reps="$reps",cfun="$cfun",rfun="$rfun",width="$width",wp="$wp" \
      run_one.sh


    echo "Submitted #$i → ${cond},${fac},${lvl},${reps},${cfun},${rfun},${width},${wp}"
    ((i++))
  else
    sleep 30
  fi
done

echo "All $TOTAL combos processed."
