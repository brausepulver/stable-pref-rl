#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "+preset.method.pref.sampler_kwargs.logging_metrics=[{_target_: pref_rl.utils.sampler.DisagreementMetric}]"
)

# Disagreement sampling
for drop_fraction in 0.2 0.05; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=disagreement" \
            "+preset.method.pref.sampler_kwargs.filters=[{_target_: pref_rl.utils.sampler.SegmentProbQuantileFilter, drop_fraction: 0.2}]" \
            "logging.tags=[pref_ppo, experiment, disagreement, policy_filtering]" \
            "logging.group=pref_ppo/policy_filtering/disagreement/${drop_fraction}"
    done
done
