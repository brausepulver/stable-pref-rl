#!/usr/bin/env bash

for seed in $(seq 0 16 144); do
    echo "Running with seed ${seed}"
    train \
        preset=ppo/quadruped_walk \
        training.total_timesteps=2000000 \
        training.seed=${seed} \
        'logging.tags=[baseline, ppo]' \
        logging.group="ppo_baseline"
done
