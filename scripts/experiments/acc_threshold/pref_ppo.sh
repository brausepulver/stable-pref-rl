#!/usr/bin/env bash

N_RUNS=56

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# Disagreement sampling
for train_acc_threshold_reward in 0.97 0.90 0.75 0.50; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.train_acc_threshold_reward=${train_acc_threshold_reward}" \
            "logging.tags=[pref_ppo, experiment, disagreement, train_acc_threshold_reward]" \
            "logging.group=pref_ppo/acc_threshold/disagreement/${train_acc_threshold_reward}"
    done
    wait
done
