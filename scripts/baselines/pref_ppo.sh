#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.unsuper.n_steps_unsuper=32000"
    "preset.method.pref.n_steps_reward=32000"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
)

# Uniform sampling
for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PrefPPO with uniform sampler and seed ${seed}"
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.sampler=uniform \
        'logging.tags=[pref_ppo, baseline, uniform]' \
        "logging.group=pref_ppo/baseline/uniform" &
done
wait

# Disagreement sampling
for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PrefPPO with disagreement sampler and seed ${seed}"
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.sampler=disagreement \
        'logging.tags=[pref_ppo, baseline, disagreement]' \
        "logging.group=pref_ppo/baseline/disagreement" &
done
wait

# Entropy sampling
for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PrefPPO with entropy sampler and seed ${seed}"
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.sampler=entropy \
        'logging.tags=[pref_ppo, baseline, entropy]' \
        "logging.group=pref_ppo/baseline/entropy" &
done
wait
