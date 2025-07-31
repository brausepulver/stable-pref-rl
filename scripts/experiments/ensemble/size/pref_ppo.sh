#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# Disagreement sampling
for ensemble_size in 5 7 10 15; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=$seed" \
            "preset.method.pref.sampler=disagreement" \
            "preset.method.pref.reward_model_kwargs.ensemble_size=${ensemble_size}" \
            "logging.tags=[pref_ppo, experiment, disagreement, ensemble, ensemble_size]" \
            "logging.group=pref_ppo/ensemble_size/${ensemble_size}"
    done
done
