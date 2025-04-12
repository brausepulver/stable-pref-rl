#!/usr/bin/env bash

for seed in 12345 23451 34512 45123 51234; do
    echo "Running DIRECTPrefPPO with seed ${seed}"

    # train \
    #     preset=direct_pref_ppo/quadruped_walk \
    #     training.total_timesteps=1000000 \
    #     training.seed=${seed} \
    #     training.num_envs=16 \
    #     preset.env.limit_ep_steps=1000 \
    #     preset.method.unsuper.n_steps_unsuper=32000 \
    #     preset.method.pref.teacher=oracle \
    #     preset.method.clip_range.end=0.3 \
    #     'logging.tags=[direct_pref_ppo, baseline]' \
    #     "logging.group=direct_pref_ppo/baseline" &

    train \
        preset=direct_pref_ppo/quadruped_walk \
        training.total_timesteps=1000000 \
        training.seed=${seed} \
        training.num_envs=16 \
        preset.env.limit_ep_steps=1000 \
        preset.method.unsuper.n_steps_unsuper=32000 \
        preset.method.pref.teacher=oracle \
        preset.method.clip_range.end=0.3 \
        'logging.tags=[direct_pref_ppo, experiment]' \
        "logging.group=direct_pref_ppo/keep_training_disc" &
done

wait
echo "All DIRECTPrefPPO runs have completed."
