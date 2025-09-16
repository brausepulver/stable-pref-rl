#!/usr/bin/env bash

N_RUNS=${1:-168}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/walker_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

# PPO
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        "${BASE_PARAMS[@]}" \
        training.seed=${seed} \
        'logging.tags=[walker_walk, ppo, baseline]' \
        logging.group=walker_walk/ppo/baseline
done

# PrefPPO
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        "${BASE_PARAMS[@]}" \
        "training.seed=${seed}" \
        "preset.method.pref.sampler=disagreement" \
        'logging.tags=[walker_walk, pref_ppo, baseline, disagreement]' \
        "logging.group=walker_walk/pref_ppo/baseline/disagreement"
done

# PrefPPO with optimal configuration
for i in $(seq 1 $N_RUNS); do
    seed=$((1000 * i))
    outb stage uv run train \
        "${BASE_PARAMS[@]}" \
        "training.seed=${seed}" \
        "preset.method.pref.sampler=disagreement" \
        "preset.method.pref.ann_buffer_size_eps=16" \
        "+preset.method.pref.margins_stats_window_size=16" \
        "+preset.method.pref.feed_buffer_size=200" \
        'logging.tags=[walker_walk, pref_ppo, optimal, disagreement]' \
        "logging.group=walker_walk/pref_ppo/optimal/disagreement"
done
