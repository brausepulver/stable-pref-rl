#!/usr/bin/env bash

for seed in 12345 23451 34512 45123 51234; do
    echo "Running PEBBLE with seed ${seed}"
    train \
        preset=pebble/quadruped_walk \
        training.total_timesteps=1000000 \
        training.seed=${seed} \
        training.num_envs=16 \
        preset.env.limit_ep_steps=1000 \
        preset.method.unsuper.n_steps_unsuper=9000 \
        preset.method.pref.teacher=oracle \
        'logging.tags=[pebble, baseline]' \
        "logging.group=pebble/baseline" &
done

wait
echo "All PEBBLE runs have completed."
