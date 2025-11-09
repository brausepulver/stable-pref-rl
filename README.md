# stable-pref-rl


## Installation

Tested with Python 3.10.

```sh
pip install -e '.[wandb,b-pref,direct]'
```

Install hyphi_gym manually due to version conflict of pinned dependency:
```sh
pip install --no-deps hyphi_gym==0.8
```


## Usage

```sh
train preset=pref_ppo/quadruped_walk
```

Other presets are available in configs/preset.

Unsupervised pre-training (see [PEBBLE](https://arxiv.org/abs/2106.05091)) is enabled by default for PrefPPO, PEBBLE and PrefDIRECT. Disable by setting `preset.method.unsuper.n_steps_unsuper=null`.


### Resume Training

Resuming training is supported for all SB3 methods and PrefPPO.

```
train \
    cfg.method._target_=<method class path> \
    load_model=<path to final_model_xyz.zip> \
    load_env=<path to vec_normalize_xyz.zip> \
    wandb_run_id=<existing run ID> \
    resume=true
```

Model and environment zipfiles are logged to the output folder of a run when training is finished or interrupted.

Passing wandb_run_id is optional. The training timestep can be reset to 0 by passing resume=false.
