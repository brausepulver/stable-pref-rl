#!/usr/bin/env bash

N_RUNS=58

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        "+preset.method.policy=SharedMlpActorCriticPolicy" \
        'logging.tags=[pref_ppo,disagreement,experiment,policy]' \
        "logging.group=pref_ppo/shared_policy"
done
