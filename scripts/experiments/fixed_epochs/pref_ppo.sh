#!/usr/bin/env bash

N_RUNS=56

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "preset.method.pref.train_acc_threshold_reward=1"
)

# Disagreement sampling
for n_epochs_reward in 1 2 5 10; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.n_epochs_reward=${n_epochs_reward}" \
            "logging.tags=[pref_ppo, experiment, disagreement, n_epochs_reward]" \
            "logging.group=pref_ppo/fixed_epochs/disagreement/${n_epochs_reward}"
    done
    wait
done
