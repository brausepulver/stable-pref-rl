#!/bin/sh

for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "Starting training run with seed $seed"
    python -m src.pref_rl.train training.seed=$seed
done
