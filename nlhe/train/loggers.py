
# nlhe/train/loggers.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from ray.tune.logger import Logger

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    from tensorboardX import SummaryWriter  # type: ignore

def _get_by_path(d: Dict[str, Any], path: str):
    """Safe nested dict lookup by 'a/b/c'. Returns None if any key is missing."""
    cur = d
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

from pprint import pprint
from numpy import float32, float64, int32, int64

class SlimTensorboardLogger(Logger):
    """
    Minimal TensorBoard logger that writes ONLY whitelisted result paths.

    Configure via class attributes BEFORE UnifiedLogger is created:
        SlimTensorboardLogger.ALLOW = ["env_runners/episode_return_mean", ...]
        SlimTensorboardLogger.TAG_MAP = {"env_runners/episode_return_mean": "reward/return_mean"}
        SlimTensorboardLogger.STEP_KEY = "training_iteration"

    You can also override these via your logger_creator prior to instantiation.
    """
    # Defaults: include common training & evaluation metrics
    ALLOW: List[str] = [
        # training
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
        "learners/default_policy/total_loss",
        # evaluation (new API)
        "evaluation/env_runners/episode_return_mean",
        "evaluation/env_runners/episode_return_max",
        "evaluation/env_runners/episode_return_min",
    ]
    TAG_MAP: Dict[str, str] = {}
    STEP_KEY: str = "training_iteration"

    def __init__(self, config: Any, logdir: str, trial: Optional[Any] = None):
        super().__init__(config, logdir, trial)
        self._writer = SummaryWriter(logdir)

        # Use class-level defaults unless overridden by logger_creator
        self._allow = list(self.ALLOW) if isinstance(self.ALLOW, list) else []
        self._tag_map = dict(self.TAG_MAP) if isinstance(self.TAG_MAP, dict) else {}
        self._step_key = str(self.STEP_KEY or "training_iteration")

        pprint(f"[SlimTensorboardLogger] logdir: {logdir}, allow={self._allow}, step_key={self._step_key}")

    def on_result(self, result: Dict[str, Any]) -> None:
        step = 0
        if isinstance(result, dict):
            v = result.get(self._step_key)
            if isinstance(v, (int, float)):
                step = int(v)

        wrote = 0
        for path in self._allow:
            val = _get_by_path(result, path)
            if val is None:
                continue
            if isinstance(val, (float, int, float32, float64, int32, int64)):
                tag = self._tag_map.get(path, path)
                self._writer.add_scalar(tag, float(val), step)
                wrote += 1
        self._writer.flush()
        # Optional debug print:
        # print(f"[SlimTensorboardLogger] step={step} wrote {wrote} scalars.")

    def close(self) -> None:
        try:
            self._writer.flush()
            self._writer.close()
        finally:
            super().close()
