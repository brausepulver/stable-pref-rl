#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.clip_range.end=0.2"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
    "preset.method.pref.sampler=entropy"
)

for n_steps_last_train in 300000 350000 400000 450000 500000; do
    echo "Running experiments with n_steps_last_train=${n_steps_last_train}"

    for i in $(seq 1 8); do
        seed=$((1000 * i))

        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            preset.method.pref.feed_batch_size=16 \
            "preset.method.pref.n_steps_reward=8000" \
            "+preset.method.pref.n_steps_last_train=${n_steps_last_train}" \
            "logging.tags=[pref_ppo, experiment, entropy, schedules, n_steps_last_train]" \
            "logging.group=pref_ppo/schedules/n_steps_8000_16_${n_steps_last_train}" &
    done
    wait
done
