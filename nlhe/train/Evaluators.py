
# nlhe/train/evaluators.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

def _get_by_path(d: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = d
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

class EvaluatorSpec:
    """
    Small helper to decouple evaluation config from PPO builder.

    Reads cfg.eval_settings and produces kwargs for AlgorithmConfig.evaluation(...).
    Fields (all optional, with safe defaults):
      - interval: int = 1                      # evaluate every N train() calls
      - num_episodes: int = 32                 # how many episodes per evaluation
      - deterministic: bool = True             # use explore=False
      - seed: int = 424242                     # fixed seed for reproducibility
      - parallel: bool = False                 # evaluate parallel to training
      - num_env_runners: int = 1               # 1 for strict reproducibility
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def spec(self) -> Dict[str, Any]:
        es = getattr(self.cfg, "eval_settings", None)
        interval      = getattr(es, "interval", 1)
        episodes      = getattr(es, "num_episodes", 32)
        deterministic = getattr(es, "deterministic", False)
        seed          = getattr(es, "seed", 424242)
        parallel      = getattr(es, "parallel", False)
        num_runners   = getattr(es, "num_env_runners", 1)

        env_cfg = {
            "hero_seat": self.cfg.env_settings.hero_seat,
            "bb": self.cfg.env_settings.bb,
            "sb": (self.cfg.env_settings.bb / 2),
            "seed": seed,
            "start_stack": self.cfg.env_settings.starting_stack,
            "history_len": self.cfg.env_settings.history_length,
        }

        return dict(
            evaluation_interval=interval,
            evaluation_duration=episodes,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=num_runners,
            evaluation_parallel_to_training=parallel,
            evaluation_config={
                "explore": (not deterministic),
                "seed": seed,
                "env_config": env_cfg,
            },
        )

    def apply_to(self, algo_cfg: AlgorithmConfig) -> AlgorithmConfig:
        return algo_cfg.evaluation(**self.spec())


class Evaluator:
    """
    Final evaluation helper. Runs RLlib's evaluate() on an Algorithm and extracts a single score.
    By default, we read 'evaluation/env_runners/episode_return_mean' with robust fallbacks.
    Returns: (score, eval_result_dict, last_checkpoint_path_or_None)
    """
    def __init__(self, cfg):
        es = getattr(cfg, "eval_settings", None)
        self.metric_path = getattr(es, "metric_path", "evaluation/env_runners/episode_return_mean")

    def run(self, algo: Algorithm) -> Tuple[float, Dict[str, Any], Optional[str]]:
        # try saving a last checkpoint for reproducibility (best-effort)
        ckpt_path: Optional[str] = None
        try:
            # Ray >= 2.6
            ckpt_path = algo.save().checkpoint.path  # type: ignore[attr-defined]
        except Exception:
            try:
                ckpt_path = algo.save()
            except Exception:
                ckpt_path = None

        eval_result: Dict[str, Any] = algo.evaluate()

        candidates = [
            self.metric_path,
            "evaluation/env_runners/episode_return_mean",
            "evaluation/episode_reward_mean",
            "env_runners/episode_return_mean",
        ]
        score_val: Optional[float] = None
        for p in candidates:
            v = _get_by_path(eval_result, p)
            if isinstance(v, (int, float)):
                score_val = float(v)
                break
        if score_val is None:
            # try inside dicts that look like {"default_policy": {"episode_return_mean": ...}}
            # last resort: search shallowly
            for k, v in eval_result.items():
                if isinstance(v, (int, float)) and "return" in k:
                    score_val = float(v)
                    break

        if score_val is None:
            raise RuntimeError(f"Evaluator: cannot extract score from eval_result (tried {candidates}).")

        return score_val, eval_result, ckpt_path
