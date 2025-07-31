#!/usr/bin/env bash

N_RUNS=${1:-56}

PRETRAIN_STEPS=(
    null
    8000
    16000
    64000
)

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

for n_steps in "${PRETRAIN_STEPS[@]}"; do
    # Uniform sampling
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.unsuper.n_steps_unsuper=${n_steps}" \
            "preset.method.pref.sampler=uniform" \
            "logging.tags=[pref_ppo, experiment, uniform, unsuper]" \
            "logging.group=pref_ppo/unsuper/${n_steps}/uniform"
    done
    
    # Disagreement sampling
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            training.seed=$seed \
            "preset.method.unsuper.n_steps_unsuper=${n_steps}" \
            "preset.method.pref.sampler=disagreement" \
            "logging.tags=[pref_ppo, experiment, disagreement, unsuper]" \
            "logging.group=pref_ppo/unsuper/${n_steps}/disagreement"
    done
done
