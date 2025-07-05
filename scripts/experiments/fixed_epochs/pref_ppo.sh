#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.unsuper.n_steps_unsuper=32000"
    "preset.method.pref.n_steps_reward=32000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
    "preset.method.pref.sampler=disagreement"
)

for n_epochs_reward in 1 2 5 10; do
    echo "Running experiments with n_epochs_reward=${n_epochs_reward}"

    for i in $(seq 1 8); do
        seed=$((1000 * i))

        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.train_acc_threshold_reward=1" \
            "preset.method.pref.n_epochs_reward=${n_epochs_reward}" \
            "logging.tags=[pref_ppo, experiment, disagreement, n_epochs_reward]" \
            "logging.group=pref_ppo/disjoint_data/disagreement/${n_epochs_reward}" &
    done
    wait
done
    