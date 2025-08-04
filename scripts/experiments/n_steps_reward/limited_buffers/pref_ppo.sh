#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "+preset.method.pref.sampler_kwargs.logging_metrics=[{_target_: pref_rl.utils.sampler.DisagreementMetric}]"
)

feed_buffer_size=200
ann_buffer_size_eps=16

# Disagreement sampling
for n_steps_reward in 48000 64000 96000 128000; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=disagreement" \
            "preset.method.pref.ann_buffer_size_eps=${ann_buffer_size_eps}" \
            "+preset.method.pref.margins_stats_window_size=${ann_buffer_size_eps}" \
            "+preset.method.pref.feed_buffer_size=${feed_buffer_size}" \
            "preset.method.pref.schedule.n_steps_reward=${n_steps_reward}" \
            "logging.tags=[pref_ppo, experiment, disagreement, ann_buffer_size_eps, feed_buffer_size, n_steps_reward]" \
            "logging.group=pref_ppo/ann_buf_size/disagreement/${ann_buffer_size_eps}/${feed_buffer_size}/${n_steps_reward}"
    done
done
