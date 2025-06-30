# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
pip install -e '.[wandb,b-pref,direct]'
pip install --no-deps hyphi_gym==0.8
```

### Training
```bash
train preset=pref_ppo/quadruped_walk
```

### Available Presets
- `pref_ppo/quadruped_walk` - Preference-based PPO with quadruped walking
- `pebble/quadruped_walk` - PEBBLE method
- `direct_pref_ppo/quadruped_walk` - Direct preference PPO
- `direct_pebble/quadruped_walk` - Direct PEBBLE
- `ppo/quadruped_walk` - Standard PPO
- `sac/quadruped_walk` - SAC baseline

### Running Experiments
Scripts are located in `scripts/` directory:
- `scripts/baselines/` - Baseline experiments
- `scripts/experiments/` - Specific experiment configurations
- `scripts/methods/` - Method-specific scripts

## Architecture

### Core Structure
- `src/pref_rl/train.py` - Main training entry point using Hydra configuration
- `src/pref_rl/config.py` - Schedule classes (Constant, Linear, Exponential, PiecewiseConstant)
- `src/pref_rl/envs.py` - Environment creation utilities

### Methods
Located in `src/pref_rl/methods/`:
- `pref_ppo.py` - Preference-based PPO implementation
- `pebble.py` - PEBBLE method
- `direct.py` - Direct preference learning
- `direct_pebble.py` - Direct PEBBLE combination
- `direct_pref_ppo.py` - Direct preference PPO combination

### Callbacks
Located in `src/pref_rl/callbacks/`:
- `pref_ppo.py` - PrefPPO-specific callbacks
- `pref.py` - General preference learning callbacks
- `direct.py` - Direct method callbacks
- `discriminator.py` - Discriminator-related callbacks
- `reward_mod.py` - Reward modification callbacks
- `unsupervised.py` - Unsupervised learning callbacks

### Utilities
Located in `src/pref_rl/utils/`:
- `logging.py` - Logging utilities
- `model.py` - Model utilities
- `pref.py` - Preference learning utilities
- `callbacks.py` - General callback utilities

### Configuration System
Uses Hydra for configuration management:
- `configs/config.yaml` - Main configuration
- `configs/preset/` - Method-specific configurations organized by method type
- `configs/envs/` - Environment-specific configurations

## Experimental Setup

### Default Training Configuration
- **Environment**: DeepMind Control Suite quadruped_walk
- **Episode Length**: 500 steps (configurable via `preset.env.limit_ep_steps`)
- **Total Training Steps**: 1M steps (configurable via `training.total_timesteps`)
- **Batch Size**: Typically 500 (determined by num_envs * steps_per_env)
- **Segment Size**: 50 steps for preference queries

### Reward Model Architecture
- **Ensemble Size**: 3 identical members (configurable via `reward_model_kwargs.ensemble_size`)
- **Network Architecture**: [256, 256, 256] (configurable via `reward_model_kwargs.net_arch`)
- **Prediction**: Mean over ensemble member predictions
- **Preference Model**: Bradley-Terry model using sigmoid over reward sums
- **Loss Function**: Cross-entropy loss on binary preferences
- **Training Target**: 97% accuracy on training data (configurable via `train_acc_threshold_reward`)

### Feedback Schedule
- **Frequency**: Every 32,000 steps (configurable via `n_steps_reward`)
- **Feedback Preference Pairs per Round**: 200 preference pairs (configurable via `feed_batch_size`)
- **Maximum Total Feedback**: 2,000 preference pairs (configurable via `max_feed`)
- **Segment Sampling**: Entropy-based sampling (preferred over disagreement or uniform)
- **Source Episodes**: Last 100 episodes for segment sampling

### Unsupervised Pre-training
- **Duration**: 32,000 steps (configurable via `n_steps_unsuper`)
- **Reward Signal**: Observation entropy
- **Purpose**: Initialize policy before reward model training
- **Disable**: Set `preset.method.unsuper.n_steps_unsuper=null`

## Key Features

### Preference Learning Methods
- **PrefPPO**: PPO with reward model trained on preferences
- **PEBBLE**: Preference-based ensemble learning
- **Sampling Strategies**: Entropy (recommended), disagreement, uniform

### Training Process
1. **Unsupervised pre-training** (32k steps with entropy reward)
2. **Iterative preference learning**:
   - Collect episodes using current policy
   - Sample segments based on entropy
   - Query preferences from teacher
   - Train reward model on preference data
   - Continue policy training with reward model
3. **Pure policy learning** (continue training policy with fixed reward model)

### Monitoring and Evaluation
- **Wandb Integration**: Comprehensive logging of training metrics
- **Key Metrics**:
  - `eval/mean_reward` - Ground-truth performance
  - `reward_model/train/accuracy` - Reward model training progress
  - `pref/num_feed` - Total preference queries used
  - `pref/ep_pred_rew_mean` - Predicted episode rewards
- **Evaluation Frequency**: Every 50k steps (configurable via `logging.eval_freq`)

### Output Structure
Training outputs in timestamped `outputs/` directories:
- `final_model_<id>.zip` - Final trained model
- `vec_normalize_<id>.pkl` - Environment normalization statistics
- `train.log` - Training logs
- `done_eps.pkl` - Episode completion tracking

## Implementation Notes

- Based on "B-Pref: Benchmarking Preference-Based Reinforcement Learning" (Lee et al, 2021)
- Uses Stable-Baselines3 for PPO implementation
- Supports MetaWorld environments with b-pref installation
- Designed for research on query efficiency and feedback schedules