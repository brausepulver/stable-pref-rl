#!/usr/bin/env bash

N_RUNS=56

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "preset.method.pref.sampler=uniform"
    "preset.method.pref.n_steps_reward=16000"
    "preset.method.pref.ann_buffer_size_eps=16"
    "+preset.method.pref.margins_stats_window_size=16"
    "preset.method.pref.max_feed=999999"
    "logging.tags=[pref_ppo,experiment,uniform,complete_feed]"
)

# Fixed accuracy threshold
for feed_batch_size in 200 400 800 1600; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))
        outb stage \
            uv run train \
                ${BASE_PARAMS[@]} \
                training.seed=$seed \
                "preset.method.pref.feed_batch_size=${feed_batch_size}" \
                "+preset.method.pref.feed_buffer_size=${feed_batch_size}" \
                "logging.group=pref_ppo/complete_feed/acc/${feed_batch_size}"
    done
done

# Fixed epochs
for feed_batch_size in 200 400 800 1600; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))
        outb stage \
            uv run train \
                ${BASE_PARAMS[@]} \
                training.seed=$seed \
                "preset.method.pref.feed_batch_size=${feed_batch_size}" \
                "+preset.method.pref.feed_buffer_size=${feed_batch_size}" \
                "preset.method.pref.n_epochs_reward=1" \
                "preset.method.pref.train_acc_threshold_reward=1" \
                "logging.group=pref_ppo/complete_feed/epochs/${feed_batch_size}"
    done
done
