# nlhe/train/tb_filter_logger.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

# Ray Tune 的 Logger 基类（RLlib 的 UnifiedLogger 也是用它）
from ray.tune.logger import Logger

# 任选其一：有 PyTorch 就用 torch 的 SummaryWriter；否则退回 tensorboardX
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    from tensorboardX import SummaryWriter  # type: ignore

def _get_by_path(d: Dict[str, Any], path: str):
    """
    在嵌套 dict 中用 'a/b/c' 路径取值；不存在返回 None。
    """
    cur = d
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Key '{k}' not found in path '{path}'")
        cur = cur[k]
    return cur

from pprint import pprint

class SlimTensorboardLogger(Logger):
    """
    只把白名单中的若干 result-路径写入 TensorBoard 的极简 Logger。

    用法（重点）：
      1) 在创建 UnifiedLogger 之前设置：
           SlimTensorboardLogger.ALLOW = ["env_runners/episode_return_mean", ...]
           SlimTensorboardLogger.TAG_MAP = {"env_runners/episode_return_mean": "reward/return_mean"}
           SlimTensorboardLogger.STEP_KEY = "training_iteration"  # 或自定义
      2) UnifiedLogger(..., loggers=(SlimTensorboardLogger,))   # 只装这一个

    好处：
      - 不会把“全量 result”落盘到 tfevents，只写白名单。
      - 不额外生成 JSON/CSV（除非你想要，可以再加对应 Logger）。
      
    """
    # tracked_metrics = {
    #         "iterations_since_restore": iterations_since_restore,
    #         "reward_mean": result["env_runners"]["episode_return_mean"],
    #         "reward_max": result["env_runners"]["episode_return_max"],
    #         "reward_min": result["env_runners"]["episode_return_min"],
    #         "episode_len_mean": result["env_runners"]["episode_len_mean"],
    #         "episode_len_max": result["env_runners"]["episode_len_max"],
    #         "episode_len_min": result["env_runners"]["episode_len_min"],
    #         "mean_kl_loss": result["learners"]["default_policy"]["mean_kl_loss"],
    #         "policy_loss": result["learners"]["default_policy"]["policy_loss"],
    #         "vf_loss": result["learners"]["default_policy"]["vf_loss"],
    #         "vf_loss_unclipped": result["learners"]["default_policy"]["vf_loss_unclipped"],
    #         "total_loss": result["learners"]["default_policy"]["total_loss"]
    #     }
    ALLOW: List[str] = [
        "env_runners/episode_return_mean",
        "env_runners/episode_return_max",
        "env_runners/episode_return_min",
        "env_runners/episode_len_mean",
        "env_runners/episode_len_max",
        "env_runners/episode_len_min",
        "learners/default_policy/mean_kl_loss",
        "learners/default_policy/policy_loss",
        "learners/default_policy/vf_loss",
        "learners/default_policy/vf_loss_unclipped",
        "learners/default_policy/total_loss"
    ]          # 允许写入的 result 路径列表（必填）
    TAG_MAP: Dict[str, str] = {}   # 可选：路径 -> TensorBoard 标签 映射
    STEP_KEY: str = "training_iteration"  # 取 step 的键（可改为 algo.iteration）

    def __init__(self, config: Any, logdir: str, trial: Optional[Any] = None):
        super().__init__(config, logdir, trial)
        self._writer = SummaryWriter(logdir)
        self.ALLOW = [
        "env_runners/episode_return_mean",
        "env_runners/episode_return_max",
        "env_runners/episode_return_min",
        "env_runners/episode_len_mean",
        "env_runners/episode_len_max",
        "env_runners/episode_len_min",
        "learners/default_policy/mean_kl_loss",
        "learners/default_policy/policy_loss",
        "learners/default_policy/vf_loss",
        "learners/default_policy/vf_loss_unclipped",
        "learners/default_policy/total_loss"
    ]  
        
        pprint(f"[SlimTensorboardLogger] logdir: {logdir}, Allow: {self.ALLOW}, Step key: {self.STEP_KEY}")

        # 兜底防呆：如果没配置白名单，就不写任何东西，避免“全量写出”
        self._allow = list(self.ALLOW) if isinstance(self.ALLOW, list) else []
        self._tag_map = dict(self.TAG_MAP) if isinstance(self.TAG_MAP, dict) else {}
        self._step_key = str(self.STEP_KEY or "training_iteration")

    def on_result(self, result: Dict[str, Any]) -> None:
        # 取 step：优先 result[STEP_KEY]，否则 0
        step = 0
        if isinstance(result, dict):
            v = result.get(self._step_key)
            if isinstance(v, (int, float)):
                step = int(v)

        # 只写白名单路径
        for path in self._allow:
            val = _get_by_path(result, path)
            if val is None:
                continue
            # 只写标量；其他类型可按需扩展（直方图、分布等）
            from numpy import float32, float64, int32, int64
            if isinstance(val, (float, int, float32, float64, int32, int64)):
                tag = self._tag_map.get(path, path)
                self._writer.add_scalar(tag, float(val), step)
            else:
                print(f"[SlimTensorboardLogger] Skipping non-scalar path '{path}' with value: {val} ({type(val)})")

        # 及时 flush（安全但略有开销；你也可每 N 次再 flush）
        self._writer.flush()
        
        print(f"[SlimTensorboardLogger] step={step} wrote {len(self._allow)} scalars.")

    def close(self) -> None:
        try:
            self._writer.flush()
            self._writer.close()
        finally:
            super().close()
