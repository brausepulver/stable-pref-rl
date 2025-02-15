#!/usr/bin/env bash

for seed in $(seq 0 16 144); do
    echo "Running with seed ${seed}"
    train \
        preset=pref_ppo/quadruped_walk \
        preset.method.unsuper.n_steps_unsuper=0 \
        preset.method.pref.n_steps_reward=32000 \
        preset.method.pref.max_feed=2000 \
        preset.method.pref.sampler=uniform \
        training.total_timesteps=4000000 \
        training.seed=${seed} \
        'logging.tags=[pref_ppo, baseline]' \
        'logging.group=pref_ppo_baseline' &
done

wait
echo "All runs have completed."
