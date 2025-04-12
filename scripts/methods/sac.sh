#!/usr/bin/env bash

for seed in 12345 23451 34512 45123 51234; do
    echo "Running SAC with seed ${seed}"
    train \
        preset=sac/quadruped_walk \
        training.total_timesteps=1000000 \
        training.seed=${seed} \
        training.num_envs=16 \
        preset.env.limit_ep_steps=1000 \
        preset.method.pref.teacher=oracle \
        'logging.tags=[sac, baseline]' \
        "logging.group=sac/baseline" &
done

wait
echo "All SAC runs have completed."
