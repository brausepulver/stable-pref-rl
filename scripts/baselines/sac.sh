#!/usr/bin/env bash

for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running SAC with seed ${seed}"
    train \
        preset=sac/quadruped_walk \
        training.total_timesteps=2000000 \
        training.seed=${seed} \
        'logging.tags=[sac, baseline]' \
        logging.group=sac/baseline &
done
wait
