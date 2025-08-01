#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "preset.method.pref.validate_on_train=false"
    "preset.method.pref.validate_on_current=false"
    "preset.method.pref.validate_on_held_out=false"
)

loss_weight=1

for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        "${BASE_PARAMS[@]}" \
        "training.seed=${seed}" \
        "preset.method.pref.sampler={_target_: pref_rl.utils.sampler.PredictedAgeMetric}" \
        "+preset.method.pref.reward_model_kind=multi_head_reward_model" \
        "+preset.method.pref.recency_loss_weight=${loss_weight}" \
        "+preset.method.pref.recency_loss_type=relative" \
        "+preset.method.pref.recency_loss_coupled=false" \
        "+preset.method.pref.recency_freeze_pref_heads=true" \
        'logging.tags=[pref_ppo, experiment, pred_age, disagreement]' \
        "logging.group=pref_ppo/pred_age/relative_loss/uncoupled/frozen/sampler"
done
