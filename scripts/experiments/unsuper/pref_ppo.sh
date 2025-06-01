#!/usr/bin/env bash

PRETRAIN_STEPS=(
    null
    8000
    16000
    32000
    64000
)

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=1000000"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.clip_range.end=0.3"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
)

for n_steps in "${PRETRAIN_STEPS[@]}"; do
    echo "Running experiments with n_steps_unsuper=${n_steps}"
    
    # Uniform sampling
    for i in $(seq 1 8); do
        seed=$((1000 * i))
        echo "Running PrefPPO with uniform sampler, n_steps_unsuper=${n_steps}, and seed ${seed}"
        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            preset.method.unsuper.n_steps_unsuper=$n_steps \
            preset.method.pref.sampler=uniform \
            "logging.tags=[pref_ppo, experiment, uniform, unsuper]" \
            "logging.group=pref_ppo/unsuper_${n_steps}/uniform" &
    done
    wait
    
    # Disagreement sampling
    for i in $(seq 1 8); do
        seed=$((1000 * i))
        echo "Running PrefPPO with disagreement sampler, n_steps_unsuper=${n_steps}, and seed ${seed}"
        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            preset.method.unsuper.n_steps_unsuper=$n_steps \
            preset.method.pref.sampler=disagreement \
            "logging.tags=[pref_ppo, experiment, disagreement, unsuper]" \
            "logging.group=pref_ppo/unsuper_${n_steps}/disagreement" &
    done
    wait
    
    # Entropy sampling
    for i in $(seq 1 8); do
        seed=$((1000 * i))
        echo "Running PrefPPO with entropy sampler, n_steps_unsuper=${n_steps}, and seed ${seed}"
        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            preset.method.unsuper.n_steps_unsuper=$n_steps \
            preset.method.pref.sampler=entropy \
            "logging.tags=[pref_ppo, experiment, entropy, unsuper]" \
            "logging.group=pref_ppo/unsuper_${n_steps}/entropy" &
    done
    wait
done

echo "All experiments completed"
