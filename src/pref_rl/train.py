import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from .envs import make_vec_env as make_env
from .methods import METHOD_DICT


def setup_wandb(cfg: DictConfig):
    run = wandb.init(
        project=cfg.logging.project,
        tags=cfg.logging.tags,
        config=dict(cfg),
        sync_tensorboard=True,
    )
    return run


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Setup WandB
    run = setup_wandb(cfg)

    # Set random seed
    set_random_seed(cfg.training.seed)

    # Create environment
    env = make_env(cfg.training.seed, n_envs=cfg.training.num_envs, name=cfg.env.name)

    # Create eval environment
    eval_env = make_env(cfg.training.seed, n_envs=1, name=cfg.env.name)

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
    method = METHOD_DICT[cfg.method.name]
    del cfg.method.name
    model = method(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        **OmegaConf.to_container(cfg.method, resolve=True)
    )

    try:
        # Train model
        model.learn(
            total_timesteps=cfg.training.total_timesteps,
            callback=[eval_callback, wandb_callback],
            progress_bar=True
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
