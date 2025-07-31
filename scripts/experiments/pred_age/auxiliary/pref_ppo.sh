#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# Disagreement sampling
for loss_weight in 0.1 0.5 1.0; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))
        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=disagreement" \
            "+preset.method.pref.num_samples_ep_age=64" \
            "+preset.method.pref.ep_age_loss_weight=${loss_weight}" \
            "+preset.method.pref.reward_model_kind=multi_head_reward_model" \
            'logging.tags=[pref_ppo, experiment, pred_age, disagreement]' \
            "logging.group=pref_ppo/pred_age/auxiliary/disagreement"
    done
done
