# pref_rl

## Installation

Tested with Python 3.10.

```sh
pip -e '.[wandb,b-pref,direct]'
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
