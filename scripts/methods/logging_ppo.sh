#!/usr/bin/env bash

for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PPO with seed ${seed}"
    train \
        preset=ppo/quadruped_walk \
        training.seed=${seed} \
        training.total_timesteps=2000000 \
        preset.method.clip_range.end=0.2 \
        preset.method._target_=pref_rl.methods.logging_ppo.LoggingPPO \
        +preset.method.save_final_ep_buffer=true \
        +preset.method.ann_buffer_size_eps=null \
        'logging.tags=[ppo, baseline]' \
        logging.group=ppo/baseline &
done
wait
