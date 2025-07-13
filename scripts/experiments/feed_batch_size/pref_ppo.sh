#!/usr/bin/env bash

N_RUNS=58

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# Disagreement sampling
for feed_batch_size in 50 100 150 300 400 500; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            preset.method.pref.feed_batch_size=$feed_batch_size \
            "logging.tags=[pref_ppo, experiment, disagreement, schedules]" \
            "logging.group=pref_ppo/schedules/feed_batch_size_$feed_batch_size"
    done
    wait
done
