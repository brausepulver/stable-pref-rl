#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

for feed_buffer_size in 200 500 1000 2000; do
    # Uniform sampling
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))
        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=uniform" \
            "+preset.method.pref.feed_buffer_size=${feed_buffer_size}" \
            "logging.tags=[pref_ppo, experiment, uniform, feed_buffer_size]" \
            "logging.group=pref_ppo/feed_buffer_size/uniform/${feed_buffer_size}"
    done

    # Disagreement sampling
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))
        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=disagreement" \
            "+preset.method.pref.feed_buffer_size=${feed_buffer_size}" \
            "logging.tags=[pref_ppo, experiment, disagreement, feed_buffer_size]" \
            "logging.group=pref_ppo/feed_buffer_size/disagreement/${feed_buffer_size}"
    done
done
