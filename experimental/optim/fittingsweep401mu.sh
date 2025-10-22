#!/usr/bin/env bash
set -euo pipefail

# ---- sweep config ----
# participants to run (edit as you like)
# participants=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
participants=({1..24})

# random seeds to run (edit as you like)
seeds=(1 )

# concurrency
max_concurrent=3
concurrents=0

# bookkeeping
id=1
outdirroot="results/mem_upd_try2/"
mkdir -p "$outdirroot"

for part in "${participants[@]}"; do
  for seed in "${seeds[@]}"; do
    outdir="${outdirroot}/run_participant${part}_seed${seed}"
    # mkdir -p "$outdir"

    echo ">>> Launching run #$id participant=${part}, seed=${seed} -> $outdir"

    python optimise_clnn.py \
      --data ~/frogs_project/data/COIN_data/trial_data_memory_updating_participant"${part}".csv \
      --out-dir "${outdir}/" \
      --klmethod analytical \
      --bs 512 \
      --save-matrices-every 1000 \
      --enable-kl-grad \
      --max-iter 10000 \
      --scale-cholesky \
      --lr 0.001 \
      --assume-opt-output-noise \
      --cuda-index 0 \
      --paradigm sr \
      --load-ys-from-file \
      --enable-qlpf \
      --model-tie-lr-weight-decay \
      --seed "${seed}" \
      &

    pid_last=$!
    echo "$pid_last" >> "$outdirroot/pids.txt"

    concurrents=$((concurrents + 1))
    if (( concurrents >= max_concurrent )); then
      wait -n
      concurrents=$((concurrents - 1))
    fi

    id=$((id + 1))
  done
done

wait  # wait for all background jobs
