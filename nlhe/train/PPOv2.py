

# checkpoints_dir = pathlib.Path("./checkpointsPRO").resolve()
# print("Checkpoints dir:", checkpoints_dir)
# checkpoints = list(checkpoints_dir.glob("checkpoint*"))
# pprint(f"total checkpoints found: {len(checkpoints)}")


# from ray.rllib.algorithms.algorithm import Algorithm

# newest_checkpoint = pathlib.Path(max(checkpoints, key=lambda f: f.stat().st_mtime)).resolve().as_posix()
# print("Loading from newest checkpoint:", newest_checkpoint)
# algo=Algorithm.from_checkpoint(newest_checkpoint)
# pprint(algo)
# config=algo.config

# print("Config:")
# pprint(config)

# import datetime
# curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# from torch.utils import tensorboard
# mkdir = "./trainlogsPPO"
# os.makedirs(mkdir, exist_ok=True)

# with tensorboard.SummaryWriter(mkdir) as tb_writer:
#     for i in range(100):
#         result = algo.train()
#         iterations_since_restore = result["iterations_since_restore"]
#         tracked_metrics = {
#             "iterations_since_restore": iterations_since_restore,
#             "reward_mean": result["env_runners"]["episode_return_mean"],
#             "reward_max": result["env_runners"]["episode_return_max"],
#             "reward_min": result["env_runners"]["episode_return_min"],
#             "episode_len_mean": result["env_runners"]["episode_len_mean"],
#             "episode_len_max": result["env_runners"]["episode_len_max"],
#             "episode_len_min": result["env_runners"]["episode_len_min"],
#             "mean_kl_loss": result["learners"]["default_policy"]["mean_kl_loss"],
#             "policy_loss": result["learners"]["default_policy"]["policy_loss"],
#             "vf_loss": result["learners"]["default_policy"]["vf_loss"],
#             "vf_loss_unclipped": result["learners"]["default_policy"]["vf_loss_unclipped"],
#             "total_loss": result["learners"]["default_policy"]["total_loss"]
#         }
        
#         # for key, value in tracked_metrics.items():
#         #     tb_writer.add_scalar(f"metrics/{key}", value, num_episodes)
#         # tb_writer.flush()
#         print(tracked_metrics)
#         if i % 10 == 0:
#             savepoint = pathlib.Path(os.path.join(checkpoints_dir, f"checkpoint_{curr_time}_iter{iterations_since_restore}.pt"))
#             print(f"Saving checkpoint to {savepoint}")
#             algo.save_checkpoint(savepoint.as_posix())

#             with open(os.path.join(mkdir, f"trainlog_{curr_time}_iter{iterations_since_restore}.txt"), "w") as f:
#                 pprint(result, stream=f)


#---file: nlhe/train/PPOv2.py

import os, sys,pathlib
from pprint import pprint
from functools import partial
from typing import Type
import time, os

import hydra

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger


from ..envs.param_env import NLHEParamEnv as NLHEGymParamEnv
from .callbacks import DefaultCallback
from .loggers import SlimTensorboardLogger   # 新增文件

def build_algo(cfg) -> Algorithm:
    """
    Build and return an algorithm instance based on the configuration.
    
    Args:
        cfg: Configuration object containing training settings with an 'algo' attribute
             that specifies which algorithm to build.
             
    Returns:
        Algorithm (Algorithm): An instance of the specified algorithm.
        
    Raises:
        NotImplementedError: If the specified algorithm is not implemented.
        
    Note:
        Currently only supports "PPO" algorithm. Other algorithms will raise
        NotImplementedError.
    """
    
    if cfg.train_settings.algo=="PPO":
        return build_ppo(cfg)
    else:
        raise NotImplementedError(f"Algorithm {cfg.train_settings.algo} not implemented")
    

def build_ppo(cfg) -> PPO:
    """
    Build and configure a PPO (Proximal Policy Optimization) algorithm using Ray RLlib.
    This function creates a PPO algorithm instance with custom configuration settings
    for training, environment, network architecture, and learning parameters specifically
    designed for No Limit Hold'em (NLHE) poker environments.
    
    Args:
        cfg: Configuration object containing the following nested settings:
        
            - train_settings: Training hyperparameters including learning rate, gamma,
              gradient clipping, number of epochs, batch sizes, and shuffling options
            - network_settings: Neural network architecture parameters including hidden
              layer sizes, activation functions, LSTM settings, and log standard deviation
            - env_settings: Environment configuration including hero seat position,
              blinds, seed, starting stack, and history length
            - log_settings: Logging and checkpointing configuration (not used in this function)
            
    Returns:
        PPO: A configured and built PPO algorithm instance ready for training.
        
    Note:
        - Uses single-process learner configuration to avoid NCCL issues on Windows
        - Configures GPU usage for training while maintaining single learner setup
        - Applies observation flattening connector for environment compatibility
        - Designed specifically for NLHEGymParamEnv poker environment
    """
    
    print("Building PPO algorithm with the following configuration:")
    pprint(cfg)
    # {'train_settings': {'algo': 'PPO', 'batch_size': 2048, 'num_mini_batches': 5096, 'learning_rate': 0.0003, 
    #                     'num_epochs': 10, 'clip_grad_norm': 1, 'gamma': 1.2, 'shuffle_batches': True}, 
    #  'network_settings': {'fc_hidden_sizes': [256, 256], 'fc_activation': 'relu', 'head_hidden_sizes': [128], 
    #                       'head_activation': 'relu', 'use_lstm': False, 'lstm_hidden_size': 128, 'free_log_std': True}, 
    #  'env_settings': {'name': 'nlhe', 'hero_seat': 1, 'bb': 2, 'sb': 1, 'seed': 42, 'starting_stack': 100, 
    #                   'history_length': 5}, 'log_settings': {'checkpointing': {'save_dir': './checkpoints',
    #                                                                            'save_freq': 1000, 'max_to_keep': 5}}}
    
    



    config = (
        PPOConfig()
        .training(lr=cfg.train_settings.learning_rate,
                  gamma=cfg.train_settings.gamma,
                  grad_clip=cfg.train_settings.clip_grad_norm,
                  num_epochs=cfg.train_settings.num_epochs,
                  shuffle_batch_per_epoch=cfg.train_settings.shuffle_batches,
                  train_batch_size_per_learner=cfg.train_settings.batch_size,
                  minibatch_size=cfg.train_settings.num_mini_batches
        )
        .environment(
            env=NLHEGymParamEnv,
            env_config={
                "hero_seat": cfg.env_settings.hero_seat,
                "bb": cfg.env_settings.bb,
                "sb": (cfg.env_settings.bb / 2),
                "seed": cfg.env_settings.seed,
                "start_stack": cfg.env_settings.starting_stack,
                "history_len": cfg.env_settings.history_length
            }
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
                free_log_std=cfg.network_settings.free_log_std
            )
        )
        .env_runners(
            env_to_module_connector=lambda env, spaces, device: [FlattenObservations()]
        )
        .learners(
                        num_learners=0,         # 单进程 Learner，避免 Windows 上的 NCCL
            num_gpus_per_learner=1, # 仍然使用 GPU 训练（cuda:0）
    )
        .callbacks(callbacks_class=partial(DefaultCallback, cfg=cfg))
    )
    return config.build_algo(logger_creator=_make_logger_creator(cfg))


def _make_logger_creator(cfg):
    """
    返回给 RLlib 的 logger_creator；只使用我们自定义的 SlimTensorboardLogger。
    """
    # 1) 配置白名单/重命名/step-key（从你的 Hydra 配置读取）
    #    你可以把下面这些字段放在 cfg.log_settings.tensorboard 下
    allow = getattr(cfg.log_settings.tensorboard, "allow", [])
    tag_map = getattr(cfg.log_settings.tensorboard, "tag_map", {})
    step_key = getattr(cfg.log_settings.tensorboard, "step_key", "training_iteration")

    # 设置到类属性（类级别参数，便于 UnifiedLogger 实例化时被读取）
    SlimTensorboardLogger.ALLOW = list(allow)
    SlimTensorboardLogger.TAG_MAP = dict(tag_map)
    SlimTensorboardLogger.STEP_KEY = step_key

    # 2) 生成日志根目录（可复用你现在的 E:/... 路径）
    base_dir = getattr(cfg.log_settings.tensorboard, "log_dir", "./tb_logs")
    ts = time.strftime("%Y%m%d_%H%M%S")

    def _logger_creator(config):
        # 每次运行一个独立目录，便于 TensorBoard 指向父目录聚合
        logdir = os.path.join(base_dir, f"ppo_{ts}")
        os.makedirs(logdir, exist_ok=True)
        # 只装我们这一个 Logger；不含 JSON/CSV/TBX => 不会写全量指标
        return UnifiedLogger(config, logdir, loggers=(SlimTensorboardLogger,))
    return _logger_creator




@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    algo=build_algo(cfg)
    print("Algorithm built:", algo)
    print("Algorithm built:", algo.trial_name, algo.trial_id,algo.logdir,algo.iteration)
    for i in range(cfg.train_settings.epochs):
        result = algo.train()
        print("Training result:", result["env_runners"]["episode_return_mean"], "iteration:", algo.iteration)

if __name__ == "__main__":
    main()