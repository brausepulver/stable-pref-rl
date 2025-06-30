#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000" \
    "preset.method.clip_range.end=0.2" \
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.unsuper.n_steps_unsuper=32000"
    "preset.method.pref.n_steps_reward=32000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
)

for max_feed in 500 1000 1500 2000 3000 4000 5000; do
    echo "Running experiments with max_feed=${max_feed}"

    for i in $(seq 1 8); do
        seed=$((1000 * i))

        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.n_steps_reward=32000" \
            "preset.method.pref.max_feed=${max_feed}" \
            preset.method.pref.feed_batch_size=200 \
            "logging.tags=[pref_ppo, experiment, entropy, max_feed]" \
            "logging.group=pref_ppo/max_feed/${max_feed}" &
    done
    wait
done
