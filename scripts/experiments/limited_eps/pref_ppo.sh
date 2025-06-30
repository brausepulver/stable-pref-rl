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

for ann_buffer_size_eps in null 128 64 32 16; do
    echo "Running experiments with ann_buffer_size_eps=${ann_buffer_size_eps}"

    for i in $(seq 1 8); do
        seed=$((1000 * i))

        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.ann_buffer_size_eps=${ann_buffer_size_eps}" \
            "logging.tags=[pref_ppo, experiment, disagreement, ann_buffer_size_eps]" \
            "logging.group=pref_ppo/ann_buffer_size_eps/disagreement/${ann_buffer_size_eps}" &
    done
    wait
done
