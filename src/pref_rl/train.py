from dotenv import load_dotenv
load_dotenv()
import hydra
from importlib.util import find_spec
from omegaconf import DictConfig, OmegaConf
import os
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
import uuid
from .envs import make_vec_env as make_env
import torch


if num_threads := os.getenv('TORCH_NUM_THREADS'):
    torch.set_num_threads(int(num_threads))

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../configs")
)

has_wandb = find_spec("wandb") is not None
_WANDB = None


def setup_wandb(cfg: DictConfig):
    global _WANDB
    import wandb
    _WANDB = wandb
    run = _WANDB.init(
        project=cfg.logging.project,
        group=cfg.logging.group,
        tags=cfg.logging.tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True,
    )
    if os.getenv('WANDB_LOG_CODE') == 'true':
        run.log_code(root='../../', include_fn=lambda path: path.startswith('src/') and path.endswith('.py'))
    return run


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    global _WANDB

    # Setup run
    run_id = setup_wandb(cfg).id if has_wandb else uuid.uuid4()

    # Create environment
    if 'policy' in cfg.preset.env:
        policy = cfg.preset.env.policy
        del cfg.preset.env.policy
    else:
        policy = 'MlpPolicy'

    env = make_env(n_envs=cfg.training.num_envs, **cfg.preset.env)

    # Create eval environment
    eval_env = make_env(**cfg.preset.env)

    # Setup callbacks
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path="./best_model" if cfg.logging.save_best_eval_models else None,
            log_path="./logs",
            eval_freq=cfg.logging.eval_freq,
            deterministic=True,
            render=False
        ),
        *([CheckpointCallback(
            save_freq=cfg.logging.checkpoint_freq,
            save_path="./checkpoints",
        )] if cfg.logging.save_checkpoints else []),
    ]
    if has_wandb:
        from wandb.integration.sb3 import WandbCallback
        callback = WandbCallback()
        callbacks.append(callback)

    # Create model
    model: BaseAlgorithm = hydra.utils.instantiate(
        cfg.preset.method,
        policy,
        env,
        verbose=1,
        tensorboard_log=f"runs/{run_id}",
        seed=cfg.training.seed,
        # Hydra arguments
        _convert_="all"
    )

    try:
        # Train model
        log_freq_rollouts = cfg.preset.get('logging', {}).get('log_freq_rollouts')
        model.learn(
            total_timesteps=cfg.training.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
            **({'log_interval': log_freq_rollouts} if log_freq_rollouts else {})
        )

        # Save final model
        model.save(f"final_model_{run_id}")

        # Save normalized env stats
        if isinstance(env, VecNormalize):
            env.save(f"vec_normalize_{run_id}.pkl")

    except KeyboardInterrupt:
        print("Saving model...")
        model.save(f"interrupted_model_{run_id}")

    finally:
        # Close environments
        env.close()
        eval_env.close()
        if has_wandb:
            _WANDB.finish()


if __name__ == "__main__":
    main()
