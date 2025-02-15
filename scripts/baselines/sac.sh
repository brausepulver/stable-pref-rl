#!/usr/bin/env bash

for seed in $(seq 0 16 144); do
    echo "Running with seed ${seed}"
    train \
        preset=sac/quadruped_walk \
        training.total_timesteps=4000000 \
        training.seed=${seed} \
        'logging.tags=[baseline, sac]' \
        logging.group="sac_baseline" &
done

wait
echo "All runs have completed."
