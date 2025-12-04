#!/bin/bash
#SBATCH --job-name=Trazz_Helper_Script
#SBATCH --output=submit_grid.%j.out
#SBATCH --error=submit_grid.%j.err
#SBATCH --time=500:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1

LIVELOG=/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/submit_grid.live.log

# create or truncate the live‐log
: > "$LIVELOG"

USER=$(whoami)
MAXJ=16
#This is changed for the top 10 runs with higher fold count
COMBOS=combos.txt
TOTAL=$(wc -l < "$COMBOS")
i=1

# example startup message
echo "$(date +'%Y-%m-%d %H:%M:%S') Starting submit_grid for $TOTAL combos" | tee -a "$LIVELOG"

while [ "$i" -le "$TOTAL" ]; do
  RUNNING=$(squeue -u "$USER" -h | wc -l)

  if [ "$RUNNING" -lt "$MAXJ" ]; then
    LINE=$(sed -n "${i}p" "$COMBOS")
    read cond fac lvl reps cfun rfun width wp iteration <<< "$LINE"

    # ensure per‐cond dir exists
    mkdir -p "/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/${cond}"
    #This is changed for the top 10 runs with hiher interation count
    OUT="/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/${cond}/${cond}_${fac}_${lvl}_${reps}_${cfun}_${rfun}_${width}_${wp}_${iteration}.csv"

    if [ -f "$OUT" ]; then
      msg="$(date +'%H:%M:%S') Skipping #$i → ${cond},${fac},${lvl},${reps},${cfun},${rfun},${width},${wp},${iteration} (already done)"
      echo "$msg" | tee -a "$LIVELOG"
      ((i++))
      continue
    fi

    # submit the job
    sbatch \
      --job-name="${cond}_${fac}_${lvl}_${reps}_${cfun}_${rfun}_${width}_${wp}_${iteration}" \
      --output="/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/logs/%x-%j.out" \
      --error="/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data/logs/%x-%j.err" \
      --export=ALL,cond="$cond",fac="$fac",lvl="$lvl",reps="$reps",cfun="$cfun",rfun="$rfun",width="$width",wp="$wp",iteration="$iteration" \
      run_one.sh

    msg="$(date +'%H:%M:%S') Submitted #$i → ${cond},${fac},${lvl},${reps},${cfun},${rfun},${width},${wp},${iteration}"
    echo "$msg" | tee -a "$LIVELOG"
    ((i++))
  else
    sleep 30
  fi
done

echo "$(date +'%H:%M:%S') All $TOTAL combos processed." | tee -a "$LIVELOG"
