from typing import Any, Dict, Optional
from stable_baselines3.common.logger import Logger
from stable_baselines3.common import utils


class PrefLogger(Logger):
    """
    Logger that extends SB3's Logger with wandb integration and progress tracking.
    """
    
    def __init__(self, folder: Optional[str], output_formats, wandb_run=None):
        super().__init__(folder, output_formats)
        self.wandb_run = wandb_run

        try:
            import wandb
            if wandb.run is not None:
                self.run = wandb.run
                self.run.define_metric(step_metric='pref/training_progress', name='pref/*')
                self.run.define_metric(step_metric='pref/num_feed', name='pref/*')
        except ImportError:
            pass
    
    def record_with_progress(self, metrics: Dict[str, Any], num_feed: int, training_progress: float, prefix: str = ""):
        """
        Record metrics to both standard logger and wandb with progress context.
        
        :param metrics: Dictionary of metrics to log
        :param num_feed: Current number of feedback samples
        :param training_progress: Current training progress (0.0 to 1.0)
        :param prefix: Optional prefix for wandb metric names
        """
        # Log to standard logger
        for key, value in metrics.items():
            if value is not None:
                self.record(key, value)
        
        # Log to wandb with progress context
        if self.wandb_run:
            wandb_metrics = {}
            
            # Add metrics with optional prefix
            for key, value in metrics.items():
                if value is not None:
                    wandb_key = f"{prefix}{key}" if prefix else key
                    wandb_metrics[wandb_key] = value
            
            # Add progress context
            wandb_metrics.update({
                'pref/num_feed': num_feed,
                'pref/training_progress': training_progress,
            })
            
            self.wandb_run.log(wandb_metrics)


def create_pref_logger(
    verbose: int = 0,
    tensorboard_log: Optional[str] = None,
    tb_log_name: str = "run",
    reset_num_timesteps: bool = True,
    wandb_run=None
) -> PrefLogger:
    """
    Create a preference learning logger with both SB3 and wandb integration.
    """
    base_logger = utils.configure_logger(verbose, tensorboard_log, tb_log_name, reset_num_timesteps)
    return PrefLogger(base_logger.dir, base_logger.output_formats, wandb_run)
