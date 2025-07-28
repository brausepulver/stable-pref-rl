#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# Disagreement sampling
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.sampler=disagreement \
        "preset.method.pref.synth_ratio=1" \
        "preset.method.pref.synth_buffer_size=200" \
        "preset.method.pref.synth_start_step=64000" \
        "preset.method.pref.synth_teacher_kwargs.neg_eps_until_steps=48000" \
        "preset.method.pref.synth_teacher_kwargs.pos_eps_after_eq_steps=16000" \
        'logging.tags=[pref_ppo,experiment,synthetic,disagreement]' \
        "logging.group=pref_ppo/experiment/synth/eph_1"
done
