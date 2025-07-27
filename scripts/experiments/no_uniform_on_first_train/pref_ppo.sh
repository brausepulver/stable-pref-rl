#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "preset.method.pref.sample_uniform_on_first_train=false"
)

# Uniform sampling
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.sampler=uniform \
        'logging.tags=[pref_ppo,baseline,uniform]' \
        "logging.group=pref_ppo/baseline/uniform"
done

# Disagreement sampling
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.sampler=disagreement \
        'logging.tags=[pref_ppo,baseline,disagreement]' \
        "logging.group=pref_ppo/baseline/disagreement"
done

# Entropy sampling
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.sampler=entropy \
        'logging.tags=[pref_ppo,baseline,entropy]' \
        "logging.group=pref_ppo/baseline/entropy"
done
