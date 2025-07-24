#!/usr/bin/env bash

N_RUNS=168

for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        preset=ppo/quadruped_walk \
        training.seed=${seed} \
        training.total_timesteps=2000000 \
        preset.method.clip_range.end=0.2 \
        'logging.tags=[ppo, baseline]' \
        logging.group=ppo/baseline
done
wait
