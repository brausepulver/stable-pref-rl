import hydra
from omegaconf import DictConfig, OmegaConf
import os
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from .envs import make_vec_env as make_env


CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../configs")
)


def setup_wandb(cfg: DictConfig):
    run = wandb.init(
        project=cfg.logging.project,
        group=cfg.logging.group,
        tags=cfg.logging.tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=True
    )
    return run


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Setup WandB
    run = setup_wandb(cfg)

    # Create environment
    env = make_env(n_envs=cfg.training.num_envs, **cfg.preset.env)

    # Create eval environment
    eval_env = make_env(**cfg.preset.env)

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=cfg.logging.eval_freq,
        deterministic=True,
        render=False
    )
    wandb_callback = WandbCallback()

    # Create model
    model: BaseAlgorithm = hydra.utils.instantiate(
        cfg.preset.method,
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        seed=cfg.training.seed,
        stats_window_size=env.num_envs,
        # Hydra arguments
        _convert_="all"
    )

    try:
        # Train model
        log_freq_rollouts = cfg.preset.get('logging', {}).get('log_freq_rollouts')
        model.learn(
            total_timesteps=cfg.training.total_timesteps,
            callback=CallbackList([eval_callback, wandb_callback]),
            progress_bar=True,
            **({'log_interval': log_freq_rollouts} if log_freq_rollouts else {})
        )

        # Save final model
        model.save(f"final_model_{run.id}")

        # Save normalized env stats
        if isinstance(env, VecNormalize):
            env.save(f"vec_normalize_{run.id}.pkl")

    except KeyboardInterrupt:
        print("Saving model...")
        model.save(f"interrupted_model_{run.id}")

    finally:
        # Close environments
        env.close()
        eval_env.close()
        wandb.finish()


if __name__ == "__main__":
    main()
