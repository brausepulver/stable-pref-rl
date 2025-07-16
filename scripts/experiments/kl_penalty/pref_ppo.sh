#!/usr/bin/env bash

N_RUNS=58

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# KL penalty experiment
for kl_penalty_coef in 0.01 0.05 0.1 0.2; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))
        outb stage uv run train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "+preset.method.kl_penalty_coef=${kl_penalty_coef}" \
            "logging.tags=[pref_ppo, experiment, disagreement, kl_penalty]" \
            "logging.group=pref_ppo/kl_penalty/disagreement/${kl_penalty_coef}"
    done
    wait
done
