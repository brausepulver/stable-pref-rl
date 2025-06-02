#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=1000000"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.clip_range.end=0.3"
    "preset.method.unsuper.n_steps_unsuper=32000"
    "preset.method.pref.n_steps_reward=32000"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
)

for i in $(seq 0 7); do
    seed=$((1000 * i))
    echo "Running PrefPPO with MIXED sampler (E:0.5, D:0.5) and seed ${seed}"
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        +preset.method.pref.sampler.type=mixed \
        +preset.method.pref.sampler.entropy=0.5 \
        +preset.method.pref.sampler.disagreement=0.5 \
        'logging.tags=[pref_ppo, experiment, sampler]' \
        "logging.group=pref_ppo/exp/sampler_mixed" &
done
wait

echo "All mixed sampler test runs launched."
