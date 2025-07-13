#!/usr/bin/env bash

N_RUNS=48

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# Uniform sampling
for max_feed in 1000 3000 4000 5000 6000
do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.sampler=uniform" \
            "preset.method.pref.n_steps_reward=32000" \
            "preset.method.pref.feed_batch_size=200" \
            "preset.method.pref.max_feed=${max_feed}" \
            "logging.tags=[pref_ppo, experiment, uniform, max_feed]" \
            "logging.group=pref_ppo/max_feed/uniform/${max_feed}"
    done
    wait
done

# Disagreement sampling
for max_feed in 1000 3000 4000 5000 6000; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.sampler=disagreement" \
            "preset.method.pref.n_steps_reward=32000" \
            "preset.method.pref.feed_batch_size=200" \
            "preset.method.pref.max_feed=${max_feed}" \
            "logging.tags=[pref_ppo, experiment, disagreement, max_feed]" \
            "logging.group=pref_ppo/max_feed/disagreement/${max_feed}"
    done
    wait
done

# Entropy sampling
for max_feed in 1000 3000 4000 5000 6000
do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            ${BASE_PARAMS[@]} \
            training.seed=$seed \
            "preset.method.pref.sampler=entropy" \
            "preset.method.pref.n_steps_reward=32000" \
            "preset.method.pref.feed_batch_size=200" \
            "preset.method.pref.max_feed=${max_feed}" \
            "logging.tags=[pref_ppo, experiment, entropy, max_feed]" \
            "logging.group=pref_ppo/max_feed/entropy/${max_feed}"
    done
    wait
done
