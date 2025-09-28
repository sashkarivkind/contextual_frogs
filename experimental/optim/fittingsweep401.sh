#!/usr/bin/env bash
set -euo pipefail

# vals=(0.01 0.05 0.1 0.2)
vals=(1 2 3 4 5 6 7 8)

id=9
for p1 in "${vals[@]}"; do
  # for p2 in "${vals[@]}"; do
    outdir="results/realthing401b/run${id}"
    mkdir -p "$outdir"

    # echo ">>> Launching run #$id with p1=$p1, p2=$p2 -> $outdir"
    echo ">>> Launching run #$id with p1=$p1 -> $outdir"

    python optimise_clnn.py --data  ~/frogs_project/data/COIN_data/trial_data_spontaneous_recovery_participant"$p1".csv \
      --out-dir "$outdir/"  \
      --klmethod analytical \
      --bs 512 \
      --save-matrices-every 1000 \
      --enable-kl-grad \
      --max-iter 40000  \
      --scale-cholesky \
      --lr 0.0001  \
      --assume-opt-output-noise \
      --cuda-index 0 --paradigm sr \
      &

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
  # done
done

wait   # <-- wait for all background jobs
