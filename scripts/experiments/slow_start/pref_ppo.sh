#!/usr/bin/env bash

BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.clip_range.end=0.2"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
    "preset.method.pref.sampler=entropy"
)

slow_start_until=300000

echo "Running experiments with slow start until ${slow_start_until}"

for i in $(seq 1 8); do
    seed=$((1000 * i))

    train \
        ${BASE_PARAMS[@]} \
        training.seed=$seed \
        preset.method.pref.feed_batch_size=16 \
        "preset.method.pref.n_steps_reward=8000" \
        "+preset.method.pref.feed_schedule._target_=pref_rl.config.PiecewiseConstantSchedule" \
        "+preset.method.pref.feed_schedule.pieces=[[0,16],[${slow_start_until},32]]" \
        "logging.tags=[pref_ppo, experiment, entropy, schedules, slow_start]" \
        "logging.group=pref_ppo/schedules/n_steps_8000_16_slow_start_${slow_start_until}" &
done
wait
