BASE_PARAMS=(
    "preset=pref_ppo/quadruped_walk"
    "training.total_timesteps=1000000"
    "training.num_envs=16"
    "preset.env.limit_ep_steps=1000"
    "preset.method.clip_range.end=0.3"
    "preset.method.unsuper.n_steps_unsuper=32000"
    "preset.method.pref.n_steps_reward=32000"
    "preset.method.pref.max_feed=2000"
    "preset.method.pref.teacher=oracle"
    "preset.method.pref.device=cpu"
    "preset.method.pref.sampler=entropy"
)

for i in $(seq 1 8); do
    seed=$((1000 * i))
    echo "Running PrefPPO with entropy sampler, outlier reward aggregator and seed ${seed}"
    train \
        ${BASE_PARAMS[@]} \
        "training.seed=$seed" \
        "+preset.method.pref.ensemble_agg_fn._target_=pref_rl.config.OutlierAggregator" \
        "logging.tags=[pref_ppo, experiment, ensemble_agg]" \
        "logging.group=pref_ppo/exp/agg_outlier" &
done
wait
