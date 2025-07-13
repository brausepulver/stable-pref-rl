#!/usr/bin/env bash

N_RUNS=48

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
)

for uniform_fraction in 0.1 0.2 0.3 0.5 0.7 0.9; do
    for sampler in disagreement entropy; do
        for i in $(seq 1 $N_RUNS); do
            seed=$((1000 * i))

            outb stage uv run train \
                ${BASE_PARAMS[@]} \
                training.seed=$seed \
                preset.method.pref.sampler=$sampler \
                "preset.method.pref.sampler_kwargs.uniform_fraction=$uniform_fraction" \
                "logging.tags=[pref_ppo, experiment, ${sampler}]" \
                "logging.group=pref_ppo/uniform_fraction/${sampler}/${uniform_fraction}" &
        done
        wait
    done
done
wait
