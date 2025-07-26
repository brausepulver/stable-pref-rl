#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "preset.method.pref.sampler=disagreement"
    "preset.method.pref.ensemble_disjoint_data=true"
)

for train_acc_threshold_reward in 0.97 0.90 0.75 0.50; do
    echo "Running experiments with train_acc_threshold_reward=${train_acc_threshold_reward}"

    for i in $(seq 1 8); do
        seed=$((1000 * i))

        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.train_acc_threshold_reward=${train_acc_threshold_reward}" \
            "preset.method.pref.n_epochs_reward=100" \
            "logging.tags=[pref_ppo, experiment, disagreement, disjoint_data, train_acc_threshold_reward]" \
            "logging.group=pref_ppo/disjoint_data/disagreement/${train_acc_threshold_reward}" &
    done
    wait
done

for n_epochs_reward in 1 2 5 10; do
    echo "Running experiments with n_epochs_reward=${n_epochs_reward}"

    for i in $(seq 1 8); do
        seed=$((1000 * i))

        train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.train_acc_threshold_reward=1" \
            "preset.method.pref.n_epochs_reward=${n_epochs_reward}" \
            "logging.tags=[pref_ppo, experiment, disagreement, disjoint_data, n_epochs_reward]" \
            "logging.group=pref_ppo/disjoint_data/disagreement/${n_epochs_reward}" &
    done
    wait
done
