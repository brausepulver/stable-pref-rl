#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.unsuper.n_steps_unsuper=32000"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
    "preset.method.pref.sampler=uniform"
    "preset.method.pref.n_steps_reward=16000"
    "preset.method.pref.max_feed=999999"
    "preset.method.pref.ann_buffer_size_eps=16"
    "preset.method.pref.n_epochs_reward=1"
    "preset.method.pref.train_acc_threshold_reward=0"
)

for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PrefPPO on approx. half of all feedback with seed ${seed}"
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        "preset.method.pref.feed_batch_size=800" \
        'logging.tags=[pref_ppo, experiment, uniform, complete_feedback]' \
        "logging.group=pref_ppo/complete_feedback/0.5" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PrefPPO on approx. all feedback with seed ${seed}"
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        "preset.method.pref.feed_batch_size=1600" \
        'logging.tags=[pref_ppo, experiment, uniform, complete_feedback]' \
        "logging.group=pref_ppo/complete_feedback/1.0" &
done
wait
