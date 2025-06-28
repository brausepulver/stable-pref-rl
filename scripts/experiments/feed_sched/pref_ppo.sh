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

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        "training.total_timesteps=2000000" \
        "preset.method.clip_range.end=0.2" \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=64000 \
        "logging.tags=[pref_ppo, experiment, entropy]" \
        "logging.group=pref_ppo/schedules/n_steps_64000" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        "training.total_timesteps=2000000" \
        "preset.method.clip_range.end=0.2" \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=128000 \
        "logging.tags=[pref_ppo, experiment, entropy]" \
        "logging.group=pref_ppo/schedules/n_steps_128000" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        "training.total_timesteps=2000000" \
        "preset.method.clip_range.end=0.2" \
        training.seed=$seed \
        preset.method.pref.n_steps_reward=192000 \
        "logging.tags=[pref_ppo, experiment, entropy]" \
        "logging.group=pref_ppo/schedules/n_steps_192000" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        "training.total_timesteps=2000000" \
        "preset.method.clip_range.end=0.2" \
        preset.method.pref.n_steps_reward=32000 \
        preset.method.pref.feed_batch_size=64 \
        "logging.tags=[pref_ppo, experiment, entropy, schedules]" \
        "logging.group=pref_ppo/schedules/n_steps_32000_64" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        "training.total_timesteps=2000000" \
        "preset.method.clip_range.end=0.2" \
        preset.method.pref.n_steps_reward=16000 \
        preset.method.pref.feed_batch_size=32 \
        "logging.tags=[pref_ppo, experiment, entropy, schedules]" \
        "logging.group=pref_ppo/schedules/n_steps_16000_32" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        "training.total_timesteps=2000000" \
        "preset.method.clip_range.end=0.2" \
        preset.method.pref.n_steps_reward=8000 \
        preset.method.pref.feed_batch_size=16 \
        "logging.tags=[pref_ppo, experiment, entropy, schedules]" \
        "logging.group=pref_ppo/schedules/n_steps_8000_16" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        "training.total_timesteps=2000000" \
        "preset.method.clip_range.end=0.2" \
        preset.method.pref.n_steps_reward=4000 \
        preset.method.pref.feed_batch_size=8 \
        "logging.tags=[pref_ppo, experiment, entropy, schedules]" \
        "logging.group=pref_ppo/schedules/n_steps_4000_8" &
done
wait

for i in $(seq 1 8); do
    seed=$((1000 * i))
    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        "training.total_timesteps=2000000" \
        "preset.method.clip_range.end=0.2" \
        preset.method.pref.n_steps_reward=500 \
        preset.method.pref.feed_batch_size=1 \
        "logging.tags=[pref_ppo, experiment, entropy, schedules]" \
        "logging.group=pref_ppo/schedules/n_steps_500_1" &
done
wait

echo "All experiments completed"
