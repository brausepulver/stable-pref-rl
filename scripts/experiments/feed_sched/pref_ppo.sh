#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=1000000"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.clip_range.end=0.3"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
    "preset.method.pref.sampler=entropy"
)

echo "Running experiments with different feedback schedules"

echo "Running experiments with constant schedules"

for i in $(seq 1 8); do
    seed=$((1000 * i))
    
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.feed_batch_size=64 \
        "logging.tags=[pref_ppo, experiment, entropy, schedules, constant_slow]" \
        "logging.group=pref_ppo/schedules/constant_64" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.feed_batch_size=150 \
        "logging.tags=[pref_ppo, experiment, entropy, schedules, constant_fast]" \
        "logging.group=pref_ppo/schedules/constant_150" &
done
wait

echo "Running experiments with logarithmic schedules"

for i in $(seq 1 8); do
    seed=$((1000 * i))
    
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=$DEFAULT_N_STEPS \
        "+preset.method.pref.feed_schedule._target_=pref_rl.config.ExponentialSchedule" \
        "+preset.method.pref.feed_schedule.start=250" \
        "+preset.method.pref.feed_schedule.end=0" \
        "+preset.method.pref.feed_schedule.decay=0.018" \
        "logging.tags=[pref_ppo, experiment, entropy, schedules, log]" \
        "logging.group=pref_ppo/schedules/log_250_0.018" &
done
wait

echo "Running experiments with varying n_steps_reward"

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=4000 \
        "logging.tags=[pref_ppo, experiment, entropy]" \
        "logging.group=pref_ppo/schedules/n_steps_4000" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=8000 \
        "logging.tags=[pref_ppo, experiment, entropy]" \
        "logging.group=pref_ppo/schedules/n_steps_8000" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=16000 \
        "logging.tags=[pref_ppo, experiment, entropy]" \
        "logging.group=pref_ppo/schedules/n_steps_16000" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=32000 \
        "logging.tags=[pref_ppo, experiment, entropy]" \
        "logging.group=pref_ppo/schedules/n_steps_32000" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=64000 \
        "logging.tags=[pref_ppo, experiment, entropy]" \
        "logging.group=pref_ppo/schedules/n_steps_64000" &
done
wait

echo "All experiments completed"
