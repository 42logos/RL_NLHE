# nlhe/train/evaluator.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config.AlgorithmConfig import evaluation

def _get_by_path(d: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = d
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

class Evaluator:
    """
    统一的评估器：使用 RLlib 的 evaluate() 在同一算法实例上评估若干 episode，
    返回一个标量评分（默认 evaluation/env_runners/episode_return_mean），
    同时可选返回完整 eval 结果字典和最终 checkpoint 路径。

    参数（从 cfg.eval_settings 读取，不存在就用默认）:
      - num_episodes: int = 32
      - deterministic: bool = True    # True => explore=False
      - metric_path: str = "evaluation/env_runners/episode_return_mean"
                     # 兼容旧字段时会自动回退
    """
    def __init__(self, cfg):
        es = getattr(cfg, "eval_settings", None)
        self.num_episodes = getattr(es, "num_episodes", 32)
        self.deterministic = getattr(es, "deterministic", True)
        self.metric_path = getattr(
            es, "metric_path", "evaluation/env_runners/episode_return_mean"
        )

    def run(self, algo: Algorithm) -> Tuple[float, Dict[str, Any], Optional[str]]:
        """
        执行评估并返回 (score, eval_result_dict, last_ckpt_path)

        说明：
          - 依赖你在构建算法时已经配置好 evaluation_*（见 PPOv2.py 的改动）。
          - 若未配置，evaluate() 也会工作，但建议按我们给的方式显式设置。
        """
        # 1) 先保存一个最终 checkpoint（可选，但超参搜索很实用）
        ckpt_path: Optional[str] = None
        try:
            ckpt_path = algo.save().checkpoint.path  # Ray >= 2.6 风格
        except Exception:
            # 兼容旧接口
            try:
                ckpt_path = algo.save()
            except Exception:
                ckpt_path = None  # 不阻塞评估

        # 2) 执行评估（使用 RLlib 的评估通道 & 同样的连接器/预处理）
        eval_result: Dict[str, Any] = algo.evaluate()

        # 3) 取评分（带多重回退，兼容不同 Ray 版本的 key）
        candidates = [
            self.metric_path,
            "evaluation/env_runners/episode_return_mean",  # 新 API 常见
            "evaluation/episode_reward_mean",              # 旧 API 常见
            "env_runners/episode_return_mean",             # 偶尔直接在根上
        ]
        score_val: Optional[float] = None
        for p in candidates:
            v = _get_by_path(eval_result, p)
            if isinstance(v, (int, float)):
                score_val = float(v)
                break
        if score_val is None:
            raise RuntimeError(
                f"Evaluator: cannot extract score from eval_result by {candidates}"
            )

        return score_val, eval_result, ckpt_path
