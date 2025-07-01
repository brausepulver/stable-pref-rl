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

for ensemble_size in 3 5 7; do
    echo "Running experiments with ensemble_size=${ensemble_size}"

    for i in $(seq 1 8); do
        seed=$((1000 * i))

        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.reward_model_kwargs.ensemble_size=${ensemble_size}" \
            "logging.tags=[pref_ppo, experiment, disagreement, ensemble_size]" \
            "logging.group=pref_ppo/ensemble_size/disagreement/${ensemble_size}" &
    done
    wait
done
