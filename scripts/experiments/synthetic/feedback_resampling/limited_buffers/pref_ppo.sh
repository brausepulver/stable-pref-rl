#!/usr/bin/env bash

N_RUNS=${1:-56}

BASE_PARAMS=(
    'hydra.run.dir=outputs/${oc.env:JOB_ID}'
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=2000000"
    "preset.method.clip_range.end=0.2"
    "preset.method.pref.validate_on_train=false"
    "preset.method.pref.validate_on_current=false"
    "preset.method.pref.validate_on_held_out=false"
    "logging.project=pref_rl_synthetic"
)

ann_buffer_size_eps=16
feed_batch_size=50
max_feed=500
synth_alpha=0.1

# Disagreement sampling
for alpha in 0.1; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))

        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=disagreement" \
            "preset.method.pref.ann_buffer_size_eps=${ann_buffer_size_eps}" \
            "+preset.method.pref.margins_stats_window_size=${ann_buffer_size_eps}" \
            "+preset.method.pref.feed_buffer_size=2000" \
            "preset.method.pref.schedule.max_feed=${max_feed}" \
            "preset.method.pref.schedule.feed_batch_size=${feed_batch_size}" \
            "preset.method.pref.synth_buffer_size=${feed_batch_size}" \
            "+preset.method.pref.synthesizer={_target_: pref_rl.utils.synthetic.FeedbackResamplingSynthesizer, alpha: ${alpha}}" \
            "+preset.method.pref.synth_schedule._target_=pref_rl.utils.train_schedules.BasePrefSchedule" \
            "+preset.method.pref.synth_schedule.n_steps_reward=32000" \
            "+preset.method.pref.synth_schedule.max_feed=null" \
            "+preset.method.pref.synth_schedule.feed_batch_size=${feed_batch_size}" \
            "+preset.method.pref.synth_schedule.n_steps_first_train=32000" \
            "+preset.method.pref.synth_schedule.n_steps_last_train=320000" \
            'logging.tags=[pref_ppo, synthetic, feedback_resampling, disagreement]' \
            "logging.group=pref_ppo/synthetic/feedback_resampling/max_feed_${max_feed}/batch_${feed_batch_size}/alpha_${alpha}"
    done
done
