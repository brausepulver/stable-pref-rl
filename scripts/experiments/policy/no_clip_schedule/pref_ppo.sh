#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.4"
)

# Uniform sampling
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        "${BASE_PARAMS[@]}" \
        "training.seed=${seed}" \
        "preset.method.pref.sampler=uniform" \
        "logging.tags=[pref_ppo, experiment, clip_range, uniform]" \
        "logging.group=pref_ppo/no_clip_schedule/uniform"
done

# Disagreement sampling
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        "${BASE_PARAMS[@]}" \
        "training.seed=${seed}" \
        "preset.method.pref.sampler=disagreement" \
        "logging.tags=[pref_ppo, experiment, clip_range, disagreement]" \
        "logging.group=pref_ppo/no_clip_schedule/disagreement"
done

# Entropy sampling
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        "${BASE_PARAMS[@]}" \
        "training.seed=${seed}" \
        "preset.method.pref.sampler=entropy" \
        "logging.tags=[pref_ppo, experiment, clip_range, entropy]" \
        "logging.group=pref_ppo/no_clip_schedule/entropy"
done
