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
)

# Disagreement sampling
for synth_batch_size in 50 100 200 400; do
    for i in $(seq 1 $N_RUNS); do
        seed=$((1000 * i))
        outb stage uv run train \
            "${BASE_PARAMS[@]}" \
            "training.seed=${seed}" \
            "preset.method.pref.sampler=disagreement" \
            "preset.method.pref.synth_buffer_size=${synth_batch_size}" \
            "+preset.method.pref.synthesizer={_target_: pref_rl.utils.synthetic.FeedbackResamplingSynthesizer}" \
            "+preset.method.pref.synth_schedule._target_=pref_rl.utils.train_schedules.BasePrefSchedule" \
            "+preset.method.pref.synth_schedule.n_steps_reward=32000" \
            "+preset.method.pref.synth_schedule.max_feed=null" \
            "+preset.method.pref.synth_schedule.feed_batch_size=${synth_batch_size}" \
            "+preset.method.pref.synth_schedule.n_steps_first_train=32000" \
            "+preset.method.pref.synth_schedule.n_steps_last_train=320000" \
            'logging.tags=[pref_ppo, synthetic, feedback_resampling, disagreement]' \
            "logging.group=pref_ppo/synthetic/feedback_resampling/batch_size_${synth_batch_size}"
    done
done
