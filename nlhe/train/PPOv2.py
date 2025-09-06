
# nlhe/train/PPOv2.py
import os, sys, pathlib
from pprint import pprint
from functools import partial
import time

import hydra

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger

from ..envs.param_env import NLHEParamEnv as NLHEGymParamEnv
from .callbacks import DefaultCallback
from .loggers import SlimTensorboardLogger
from .Evaluators import EvaluatorSpec, Evaluator

def build_algo(cfg):
    if cfg.train_settings.algo == "PPO":
        return build_ppo(cfg)
    else:
        raise NotImplementedError(f"Algorithm {cfg.train_settings.algo} not implemented")

def build_ppo(cfg) -> PPO:
    print("Building PPO algorithm with the following configuration:")
    pprint(cfg)

    config = (
        PPOConfig()
        .training(
            lr=cfg.train_settings.learning_rate,
            gamma=cfg.train_settings.gamma,
            grad_clip=cfg.train_settings.clip_grad_norm,
            num_epochs=cfg.train_settings.num_epochs,
            shuffle_batch_per_epoch=cfg.train_settings.shuffle_batches,
            train_batch_size_per_learner=cfg.train_settings.batch_size,
            minibatch_size=cfg.train_settings.num_mini_batches,
        )
        .environment(
            env=NLHEGymParamEnv,
            env_config={
                "hero_seat": cfg.env_settings.hero_seat,
                "bb": cfg.env_settings.bb,
                "sb": (cfg.env_settings.bb / 2),
                "seed": cfg.env_settings.seed,
                "start_stack": cfg.env_settings.starting_stack,
                "history_len": cfg.env_settings.history_length,
            },
        )
        .rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=cfg.network_settings.fc_hidden_sizes,
                fcnet_activation=cfg.network_settings.fc_activation,
                head_fcnet_hiddens=cfg.network_settings.head_hidden_sizes,
                head_fcnet_activation=cfg.network_settings.head_activation,
                use_lstm=cfg.network_settings.use_lstm,
                lstm_cell_size=cfg.network_settings.lstm_hidden_size,
                max_seq_len=cfg.env_settings.history_length,
                free_log_std=cfg.network_settings.free_log_std,
            )
        )
        .env_runners(env_to_module_connector=lambda env, spaces, device: [FlattenObservations()], num_env_runners=8)
        .learners(num_learners=0, num_gpus_per_learner=1)
        .callbacks(callbacks_class=partial(DefaultCallback, cfg=cfg))
    )

    # Apply evaluation spec produced by separate module (closure/object)
    config = EvaluatorSpec(cfg).apply_to(config)

    # Build with our slim TB logger
    try:
        algo = config.build(logger_creator=_make_logger_creator(cfg))
    except TypeError:
        # some versions use build_algo
        algo = config.build_algo(logger_creator=_make_logger_creator(cfg))  # type: ignore[attr-defined]
    return algo

def _make_logger_creator(cfg):
    # Configure whitelist/tag map/step key (from cfg.log_settings.tensorboard)
    allow = getattr(cfg.log_settings.tensorboard, "allow", [])
    tag_map = getattr(cfg.log_settings.tensorboard, "tag_map", {})
    step_key = getattr(cfg.log_settings.tensorboard, "step_key", "training_iteration")

    SlimTensorboardLogger.ALLOW = list(allow) if allow else SlimTensorboardLogger.ALLOW
    SlimTensorboardLogger.TAG_MAP = dict(tag_map) if tag_map else {}
    SlimTensorboardLogger.STEP_KEY = step_key

    base_dir = getattr(cfg.log_settings.tensorboard, "log_dir", "./tb_logs")
    ts = time.strftime("%Y%m%d_%H%M%S")

    def _logger_creator(config):
        logdir = os.path.join(base_dir, f"ppo_{ts}")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=(SlimTensorboardLogger,))
    return _logger_creator

def train_and_eval(cfg) -> float:
    """
    Train for cfg.train_settings.epochs iterations, then run a final deterministic evaluation.
    Return the single scalar score for hyperparameter selection.
    """
    algo = build_algo(cfg)
    epochs = int(cfg.train_settings.epochs)
    for _ in range(epochs):
        result = algo.train()
        # optional: read automatic evaluation results from result["evaluation"]
        # print(f"train iter {result['training_iteration']} reward {result['episode_reward_mean']}")

    evaluator = Evaluator(cfg)
    score, eval_result, ckpt_path = evaluator.run(algo)
    print(f"[FINAL EVAL] score={score} checkpoint={ckpt_path}")
    return score

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    score = train_and_eval(cfg)
    print(f"[FINAL_SCORE] {score}")
    return score

if __name__ == "__main__":
    main()
