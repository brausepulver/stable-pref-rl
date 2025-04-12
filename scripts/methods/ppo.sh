#!/usr/bin/env bash

for seed in 12345 23451 34512 45123 51234; do
    echo "Running PPO with seed ${seed}"
    train \
        preset=ppo/quadruped_walk \
        training.total_timesteps=1000000 \
        training.seed=${seed} \
        training.num_envs=16 \
        preset.env.limit_ep_steps=1000 \
        preset.method.clip_range.end=0.3 \
        'logging.tags=[ppo, baseline]' \
        "logging.group=ppo/baseline" &
done

wait
echo "All PPO runs have completed."
