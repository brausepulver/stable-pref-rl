#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=320000"
    "preset.method.clip_range.end=0.368"
    "preset.method.pref.validate_on_train=false"
    "preset.method.pref.validate_on_current=false"
    "preset.method.pref.validate_on_held_out=false"
)

# Disagreement sampling
for loss_weight in 0.1 1 10; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))
        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=disagreement" \
            "+preset.method.pref.reward_model_kind=multi_head_reward_model" \
            "+preset.method.pref.recency_loss_weight=${loss_weight}" \
            "+preset.method.pref.recency_loss_type=relative" \
            "+preset.method.pref.recency_loss_coupled=false" \
            "+preset.method.pref.recency_freeze_pref_heads=true" \
            'logging.tags=[pref_ppo, experiment, pred_age, disagreement]' \
            "logging.group=pref_ppo/pred_age/relative_loss/uncoupled/frozen/loss_only/${loss_weight}"
    done
done
