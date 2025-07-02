#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000" \
    "preset.method.clip_range.end=0.2" \
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.unsuper.n_steps_unsuper=32000"
    "preset.method.pref.n_steps_reward=32000"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
)

for uniform_fraction in 0.1 0.2 0.3 0.5 0.7 0.9; do
    for sampler in disagreement entropy; do
        for i in $(seq 1 8); do
            seed=$((1000 * i))
            echo "Running PrefPPO with ${sampler} sampler and uniform_fraction=${uniform_fraction} (seed ${seed})"
            train \
                ${BASE_PARAMS[@]} \
                training.seed=$seed \
                preset.method.pref.sampler=$sampler \
                preset.method.pref.sampler.uniform_fraction=$uniform_fraction \
                "logging.tags=[pref_ppo, experiment, ${sampler}]" \
                "logging.group=pref_ppo/uniform_fraction/${sampler}/${uniform_fraction}" &
        done
        wait
    done
done
wait
