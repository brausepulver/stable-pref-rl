#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# Disagreement sampling
for direction in 1 -1; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "+preset.method.pref.sampler={_target_: pref_rl.utils.sampler.SegmentProbMetric, direction: ${direction}}" \
            "logging.tags=[pref_ppo, experiment, disagreement, policy_sampling]" \
            "logging.group=pref_ppo/policy_sampling/disagreement/${direction}"
    done
done
