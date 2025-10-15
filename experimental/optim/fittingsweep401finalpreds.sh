#!/usr/bin/env bash
set -euo pipefail

# vals=(0.01 0.05 0.1 0.2)
# vals=(3 4 13)
vals=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# seeds=(1 2 3)
seeds=(1 ) # 2 3 4 5)

max_concurrent=5
concurrents=0
id=1
outdirroot="results/realthing403uuu_finalpredsFX/"
source_outdirroot="results/realthing403uuu_seedsFX/"
for p2 in "${seeds[@]}"; do
  for p1 in "${vals[@]}"; do
    outdir="${outdirroot}/run${p1}_seed${p2}"
    source_outdir="${source_outdirroot}/run${p1}_seed${p2}"

    if (( p1 <= 8 )); then
      recoverytype="evoked"
      subindex=$p1
    else
      recoverytype="spontaneous"
      subindex=$((p1 - 8))
    fi

    mkdir -p "$outdir"

    # echo ">>> Launching run #$id with p1=$p1, p2=$p2 -> $outdir"
    echo ">>> Launching run #$id with p1=$p1 -> $outdir"

    python optimise_clnn.py --data  ~/frogs_project/data/COIN_data/trial_data_"$recoverytype"_recovery_participant"$subindex".csv \
      --out-dir "$outdir/"  \
      --klmethod analytical \
      --bs 512 \
      --save-matrices-every 1000 \
      --enable-kl-grad \
      --max-iter 10000  \
      --scale-cholesky \
      --lr 0.0  \
      --assume-opt-output-noise \
      --cuda-index 0 --paradigm NA \
      --load-ys-from-file \
      --seed "$p2" \
      --reuse "$source_outdir/"/params.pt \
       --eval-only --paradigm-file ../../signoffrepertoire1.pkl  \
        --model-tie-lr-weight-decay \
      &
            # --save-batch-of-trajs \

      pid=$!
      echo "$pid" >> "$outdirroot/pids.txt"
      concurrents=$((concurrents + 1))
      if (( concurrents >= max_concurrent )); then
        wait -n
        concurrents=$((concurrents - 1))
      fi

    # python optimise_clnn.py \
    #   --data "$HOME/scratch/sasha-model-optimization/_data_er5/test" \
    #   --out-dir "$outdir/" \
    #   --klmethod analytical \
    #   --model toy \
    #   --bs 512 \
    #   --t-episode 100 \
    #   --save-matrices-every 1000 \
    #   --optimize-toy-noises \
    #   -tdsp "$p1" \
    #   -tdso "$p2" \
    #   --enable-kl-grad \
    #   &   # <-- run in background

    id=$((id+1))
  done
done

wait   # <-- wait for all background jobs
