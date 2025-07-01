#!/usr/bin/env bash

for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PPO with seed ${seed}"
    train \
        preset=ppo/quadruped_walk \
        training.seed=${seed} \
        training.total_timesteps=1000000 \
        preset.method.clip_range.end=0.3 \
        preset.method._target_=pref_rl.methods.logging_ppo.LoggingPPO \
        'logging.tags=[ppo, baseline]' \
        logging.group=ppo/baseline &
done
wait
