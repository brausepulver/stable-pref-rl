# pref_rl

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

Other presets are available in config/presets.

Unsupervised pre-training (see [PEBBLE](https://arxiv.org/abs/2106.05091)) is enabled by default for PrefPPO, PEBBLE and PrefDIRECT. Disable by setting `preset.method.unsuper.n_steps_unsuper=null`.
