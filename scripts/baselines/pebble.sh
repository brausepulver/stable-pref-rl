#!/usr/bin/env bash

N_RUNS=${1:-168}

BASE_PARAMS=(
    "preset=pebble/quadruped_walk"
    "training.total_timesteps=2000000"
    "training.num_envs=16"
)

for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    train \
        ${BASE_PARAMS[@]} \
        training.seed=${seed} \
        preset.method.pref.sampler=disagreement \
        "logging.group=pebble/baseline/disagreement" \
        'logging.tags=[pebble, baseline]' &
done
wait

for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    train \
        ${BASE_PARAMS[@]} \
        training.seed=${seed} \
        preset.method.pref.sampler=disagreement \
        "preset.method.pref.ann_buffer_size_eps=16" \
        "+preset.method.pref.margins_stats_window_size=16" \
        "+preset.method.pref.feed_buffer_size=200" \
        "logging.group=pebble/optimal/disagreement" \
        'logging.tags=[pebble, baseline]' &
done
wait
