#!/usr/bin/env bash

for seed in 12345 23451 34512 45123 51234; do
    echo "Running DIRECTPrefPPO with seed ${seed}"

    train \
        preset=direct_pref_ppo/quadruped_walk \
        training.total_timesteps=1000000 \
        training.seed=${seed} \
        preset.method.clip_range.end=0.3 \
        'logging.tags=[direct_pref_ppo, baseline]' \
        logging.group=direct_pref_ppo/baseline &
done
wait
