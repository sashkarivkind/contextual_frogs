#!/usr/bin/env bash
set -euo pipefail

# Concurrency (override like: JOBS=8 ./run_grid_parallel.sh)
JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"

vals=(0.01 0.05 0.1 0.2)

# Build (id, p1, p2) triples
jobs=()
id=1
for p1 in "${vals[@]}"; do
  for p2 in "${vals[@]}"; do
    jobs+=("$id $p1 $p2")
    ((id++))
  done
done

run_cmd() {
  local id="$1" p1="$2" p2="$3"
  local outdir="results/recoverysweep101a/run${id}"
  mkdir -p "$outdir"
  echo ">>> [#${id}] p1=$p1 p2=$p2 -> $outdir"

  python optimise_clnn.py \
    --data "$HOME/scratch/sasha-model-optimization/_data_er5/test" \
    --out-dir "$outdir/" \
    --klmethod analytical \
    --model toy \
    --bs 512 \
    --t-episode 100 \
    --save-matrices-every 1000 \
    --optimize-toy-noises \
    -tdsp "$p1" \
    -tdso "$p2" \
    --enable-kl-grad \
    --max-iter 20
}

if command -v parallel >/dev/null 2>&1; then
  # Fast path: GNU parallel
  printf '%s\n' "${jobs[@]}" | parallel -j "$JOBS" --colsep ' ' \
    'bash -c '\''run_cmd "$@"'\'' _ {1} {2} {3}' \
    ::::
else
  # Fallback: simple Bash worker pool
  active=0
  for line in "${jobs[@]}"; do
    # shellcheck disable=SC2086
    run_cmd $line &
    ((active++))
    if (( active >= JOBS )); then
      wait -n
      ((active--))
    fi
  done
  wait
fi

echo "All runs finished."
