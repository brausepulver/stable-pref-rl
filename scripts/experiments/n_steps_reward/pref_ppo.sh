#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# Disagreement sampling
for n_steps_reward in 4000 8000 16000 64000 96000 128000; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=disagreement" \
            "preset.method.pref.n_steps_reward=${n_steps_reward}" \
            "logging.tags=[pref_ppo, experiment, disagreement, schedules]" \
            "logging.group=pref_ppo/n_steps_reward/disagreemet/${n_steps_reward}"
    done
    wait
done
