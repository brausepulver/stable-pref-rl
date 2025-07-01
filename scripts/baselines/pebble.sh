#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pebble/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
)

for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PEBBLE with entropy sampler and seed ${seed}"
    train \
        ${BASE_PARAMS[@]} \
        training.seed=${seed} \
        preset.method.pref.sampler=entropy \
        "logging.group=pebble/baseline/entropy" \
        'logging.tags=[pebble, baseline]'&
done
wait
