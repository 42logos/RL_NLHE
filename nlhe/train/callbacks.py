# -- file: nlhe/train/callbacks.py
from __future__ import annotations
from datetime import datetime
import os
import pprint
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from gymnasium import Env
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.algorithms.algorithm import Algorithm

import yaml 

class DefaultCallback(RLlibCallback):
    """A comprehensive callback for RLlib training with automatic checkpointing and configuration management.
    This callback extends RLlibCallback to provide automated checkpoint saving during training
    with configurable frequency, directory management, and cleanup policies. It integrates
    seamlessly with RLlib's training lifecycle and handles edge cases gracefully.
    The callback supports:
    - Configurable checkpoint frequency based on training iterations
    - Automatic directory creation with timestamp-based naming
    - Configuration backup (algorithm config saved as YAML)
    - Intelligent pruning to maintain a maximum number of checkpoints
    - Robust error handling and defensive programming practices
    
    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing log_settings with
            checkpointing parameters. Expected structure:
            ```yaml
            {
                "log_settings": {
                    "checkpointing": {
                        "save_freq": int,              # Checkpoint every N iterations (<=0 disables)
                        "next_transparent_number": int, # Starting checkpoint number (default: 1)
                        "next_checkpoint_number": int,  # Alternative key for checkpoint number
                        "next_number": int,            # Fallback key for checkpoint number
                        "savingdir": str,              # Base directory for checkpoints
                        "save_dir": str,               # Alternative key for save directory
                        "max_to_keep": int,            # Maximum number of checkpoints to retain
                    }
                }
            }
            ```
                
            
    Attributes:
        frequency (int): Number of training iterations between checkpoints. Values <= 0 disable
            checkpointing entirely.
        next_num (int): Sequential number for the next checkpoint directory. Increments after
            each successful save operation.
        save_root (Path): Root directory where checkpoints are stored. Automatically includes
            a timestamp subdirectory to avoid conflicts between training runs.
        max_to_keep (int): Maximum number of checkpoint directories to retain. Older checkpoints
            are automatically pruned when this limit is exceeded.
        algo_conf (Dict): Cached copy of the algorithm configuration, saved as YAML for
            reproducibility and debugging purposes.
            
    Example:
        ```python
        config = {
            "log_settings": {
                "checkpointing": {
                    "save_freq": 100,           # Save every 100 iterations
                    "next_number": 1,           # Start numbering from 1
                    "save_dir": "./my_checkpoints",
                    "max_to_keep": 3            # Keep only 3 most recent checkpoints
                }
            }
        }
        callback = DefaultCallback(config)
        # Use with RLlib training
        algo = PPO(config=ppo_config, callbacks=[callback])
        for i in range(1000):
            result = algo.train()
            # Checkpoints automatically saved every 100 iterations
        ```
    Note:
        The callback uses defensive programming patterns to handle various edge cases:
        - Multiple configuration key variants for backward compatibility
        - Graceful handling of missing or malformed iteration numbers
        - Automatic directory creation if on_algorithm_init is skipped
        - Robust checkpoint directory naming with zero-padding for proper sorting
        Checkpoint directories follow the pattern: checkpoint_0000001, checkpoint_0000002, etc.
        Each checkpoint contains the complete algorithm state as saved by RLlib's native
        save_to_path() method.
        
    Raises:
        ValueError: When algorithm configuration is None or cannot be accessed during
            initialization phase.
        Various filesystem-related exceptions may occur during directory operations,
        but these are generally handled gracefully with appropriate logging.
    """
    

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()  # 按 RLlibCallback 规范先调用父类
        cp = (cfg.get("log_settings", {}) or {}).get("checkpointing", {}) or {}

        # 读取并规范化配置
        self.frequency: int = int(cp.get("save_freq", 0))  # <=0 表示禁用
        # 兼容“写错键名”的场景：transparent/transpose/next_number 统统兜底
        self.next_num: int = int(
            cp.get("next_transparent_number",
                   cp.get("next_checkpoint_number",
                          cp.get("next_number", 1)))
        )
        self.save_root: Path = Path(os.path.join(Path(cp.get("savingdir", cp.get("save_dir", "./checkpoints"))), datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.max_to_keep: int = int(cp.get("max_to_keep", 5))



        # 懒初始化：在 on_algorithm_init 再建目录和做一次清理
        self._initialized = False
        
        # log 相关
        self.logger=MetricsLogger()


        print(f"cp: {cp}, save_root: {self.save_root}, max_to_keep: {self.max_to_keep}")

        

    # ---- 生命周期钩子（签名严格遵循文档） ----
    def on_algorithm_init(self, *, algorithm: Algorithm, metrics_logger=None, **kwargs) -> None:
        """Initializes the callback when the algorithm is first created.
        This method is called once during the algorithm initialization phase and performs
        several setup operations including directory creation, checkpoint pruning, and
        configuration saving. It ensures the callback is properly initialized and the
        algorithm configuration is persisted to disk for reproducibility.
        
        Args:
            algorithm (Algorithm): The algorithm instance being initialized. Must have a
                valid config attribute that can be converted to a dictionary representation.
            metrics_logger (optional): Logger instance for recording metrics during training.
                Defaults to None if not provided.
            **kwargs: Additional keyword arguments that may be passed by the training
                framework. These are currently unused but maintained for compatibility.
                
        Raises:
            ValueError: If the algorithm's config attribute is None, preventing the
                configuration from being saved to disk.
                
        Side Effects:
            - Creates the save_root directory and any necessary parent directories
            - Prunes existing checkpoints to maintain only max_to_keep files
            - Sets the internal _initialized flag to True
            - Saves the algorithm configuration as a YAML file named 'algo_config.yaml'
              in the save_root directory
              
        Note:
            This method should only be called once per algorithm instance. The callback
            must be properly configured with a valid save_root path before this method
            is invoked.
        """
        
        self.save_root.mkdir(parents=True, exist_ok=True)
        self._prune_to_k(self.max_to_keep)
        self._initialized = True
        if not algorithm.config == None:
            self.algo_conf = algorithm.config.to_dict()
        else:
            raise ValueError("Algorithm config is None, cannot save configuration.")
        with open(self.save_root / "algo_config.yaml", "w") as f:
            f.write(yaml.dump(self.algo_conf))

    def on_train_result(
        self,
        *,
        algorithm,                                 # 当前 Algorithm
        result: Optional[Dict[str, Any]] = None,   # 这次 train() 返回字典
        metrics_logger=None,                       # 可记录自定义指标
        **kwargs,
    ) -> None:
        """Handle training results and perform checkpointing based on configured frequency.
        This callback method is invoked after each training iteration to evaluate whether
        a checkpoint should be saved. It implements frequency-based checkpointing with
        automatic cleanup to maintain a maximum number of stored checkpoints.
        
        Args:
            algorithm: The current training algorithm instance that contains the model
                state and training progress. Must implement the Checkpointable interface
                with save_to_path() method.
            result: Optional dictionary containing training metrics and metadata returned
                from the current train() call. Expected to contain 'training_iteration'
                key for iteration tracking. Defaults to None.
            metrics_logger: Optional logger instance for recording custom metrics during
                the checkpointing process. Defaults to None.
            **kwargs: Additional keyword arguments passed from the training framework.
                Currently unused but maintained for compatibility.
                
        Returns:
            None: This method performs side effects (saving checkpoints) but returns
                no value.
                
        Behavior:
            - Skips checkpointing if frequency is disabled (frequency <= 0)
            - Performs lazy initialization if on_algorithm_init was not called
            - Determines current iteration from result dict or algorithm attribute
            - Only saves checkpoints at iterations that are multiples of frequency
            - Creates numbered checkpoint directories (checkpoint_0000001, etc.)
            - Uses algorithm's save_to_path() method for actual state persistence
            - Automatically prunes old checkpoints to maintain max_to_keep limit
            
        Side Effects:
            - Creates checkpoint directories under self.save_root
            - Writes model state files via algorithm.save_to_path()
            - Deletes old checkpoint directories when exceeding max_to_keep
            - Prints status messages for debugging and monitoring
            - Increments internal checkpoint counter (self.next_num)
            
        Raises:
            No explicit exceptions raised, but underlying save_to_path() may raise
            IOError or other filesystem-related exceptions.
            
        Note:
            This method assumes the algorithm implements Ray RLlib's Checkpointable
            interface. The checkpoint numbering is sequential and independent of
            training iteration numbers to ensure uniqueness.
        """
        print(f"on_train_result called at iteration {getattr(algorithm, 'iteration', None)}")
        if self.frequency <= 0:
            print("Checkpointing disabled (frequency <= 0).")
            return
        if not self._initialized:
            # 防御：若没走到 on_algorithm_init（极少数情况），这里兜底建目录
            print("Warning: on_train_result called before on_algorithm_init, initializing now.")
            self.on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger)

        # 训练迭代号：优先 result["training_iteration"]，否则从 algo 读
        iter_no = None
        if isinstance(result, dict):
            iter_no = result.get("training_iteration")
        if iter_no is None and hasattr(algorithm, "iteration"):
            iter_no = getattr(algorithm, "iteration")

        if not isinstance(iter_no, int) or iter_no <= 0:
            print(f"Warning: cannot determine current iteration number, got {iter_no}")
            return
        if iter_no % self.frequency != 0:
            print(f"Skipping checkpoint at iteration {iter_no} (not multiple of {self.frequency})")
            return

        # 保存：savingdir/checkpoint_0000001 这样的编号目录
        ckpt_dir = Path(self.save_root / f"checkpoint_{self.next_num:07d}").resolve()
        self.next_num += 1

        # 官方 Checkpointable API：Algorithm.save_to_path(path=...)
        saved_path = algorithm.save_to_path(path=str(ckpt_dir))  # 返回实际写入位置
        print(f"Checkpoint saved to {saved_path} at iteration {iter_no}")
        # 超额即删最旧，确保最多 max_to_keep 个
        self._prune_to_k(self.max_to_keep)
        
        self._log_rst(result)


    

    # ---- 内部工具 ----
    def _list_ckpts(self) -> List[Tuple[int, Path]]:
        items: List[Tuple[int, Path]] = []
        for p in self.save_root.glob("checkpoint_*"):
            m = re.fullmatch(r"checkpoint_(\d+)", p.name)
            if m:
                items.append((int(m.group(1)), p))
        items.sort(key=lambda t: t[0])
        return items

    def _prune_to_k(self, k: int) -> None:
        items = self._list_ckpts()
        if len(items) > k:
            for _, oldp in items[:-k]:
                shutil.rmtree(oldp, ignore_errors=True)

    def _log_rst(self, result: Dict[str, Any]) -> None:
        tracked_metrics = {
            "reward_mean": result["env_runners"]["episode_return_mean"],
            "reward_max": result["env_runners"]["episode_return_max"],
            "reward_min": result["env_runners"]["episode_return_min"],
            "episode_len_mean": result["env_runners"]["episode_len_mean"],
            "episode_len_max": result["env_runners"]["episode_len_max"],
            "episode_len_min": result["env_runners"]["episode_len_min"],
            "mean_kl_loss": result["learners"]["default_policy"]["mean_kl_loss"],
            "policy_loss": result["learners"]["default_policy"]["policy_loss"],
            "vf_loss": result["learners"]["default_policy"]["vf_loss"],
            "vf_loss_unclipped": result["learners"]["default_policy"]["vf_loss_unclipped"],
            "total_loss": result["learners"]["default_policy"]["total_loss"]
        }
        for key, value in tracked_metrics.items():
            self.logger.log_value(f"metrics/{key}", value)
