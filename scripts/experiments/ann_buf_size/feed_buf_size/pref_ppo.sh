#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "+preset.method.pref.sampler_kwargs.logging_metrics=[{_target_: pref_rl.utils.sampler.DisagreementMetric}]"
)

for feed_buffer_size in 200 1000; do
    # Uniform sampling
    for ann_buffer_size_eps in null 64 32 16; do
        for i in $(seq 1 $N_RUNS); do
            seed=$((1000 * i))

            outb stage uv run train \
                ${BASE_PARAMS[@]} \
                training.seed=${seed} \
                "preset.method.pref.sampler=uniform" \
                "preset.method.pref.ann_buffer_size_eps=${ann_buffer_size_eps}" \
                "+preset.method.pref.margins_stats_window_size=${ann_buffer_size_eps}" \
                "+preset.method.pref.feed_buffer_size=${feed_buffer_size}" \
                "logging.tags=[pref_ppo, experiment, uniform, ann_buffer_size_eps, feed_buffer_size]" \
                "logging.group=pref_ppo/ann_buf_size/uniform/${ann_buffer_size_eps}/${feed_buffer_size}"
        done
        wait
    done

    # Disagreement sampling
    for ann_buffer_size_eps in null 64 32 16; do
        for i in $(seq 1 $N_RUNS); do
            seed=$((1000 * i))

            outb stage uv run train \
                "${BASE_PARAMS[@]}" \
                "training.seed=${seed}" \
                "preset.method.pref.sampler=disagreement" \
                "preset.method.pref.ann_buffer_size_eps=${ann_buffer_size_eps}" \
                "+preset.method.pref.margins_stats_window_size=${ann_buffer_size_eps}" \
                "+preset.method.pref.feed_buffer_size=${feed_buffer_size}" \
                "logging.tags=[pref_ppo, experiment, disagreement, ann_buffer_size_eps, feed_buffer_size]" \
                "logging.group=pref_ppo/ann_buf_size/disagreement/${ann_buffer_size_eps}/${feed_buffer_size}"
        done
        wait
    done
done
