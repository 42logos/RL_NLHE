# evaluate_preflop_grid.py
# 评估：翻前 vs 2bb 注，agent 对 13x13 起手牌网格的弃/跟/加概率
# 用法：
#   python evaluate_preflop_grid.py --ckpt /path/to/checkpoint_dir --samples-per-combo 64 --combos-per-cell 24
#
# 产物：
#   ./eval_out/
#     fold_grid.csv / call_grid.csv / raise_grid.csv
#     fold_grid.png / call_grid.png / raise_grid.png

import argparse
import os
from pathlib import Path
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from nlhe.envs.param_env import NLHEParamEnv
from ray.tune.registry import register_env
from ray.rllib.connectors.env_to_module import FlattenObservations
import torch
import tqdm
from functools import partial


from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.algorithms.algorithm import Algorithm
from nlhe.train.callbacks import DefaultCallback
from ray.rllib.algorithms.algorithm import Algorithm
# ------------------ 配置超参数 ------------------
RANKS = "AKQJT98765432"       # 13 个等级（A 在左上角）
SENTINEL = -1                 # env 中未揭示牌的占位符
DEFAULT_STACK = 100           # 英雄筹码（只用于构造观测）
SB, BB = 1, 2                 # 小盲/大盲，决定 pot 的参考值
POT_FACING_2BB = SB + BB      # 翻前未加注 pot≈3，这里简化为 3
CURRENT_BET = 2               # 面对 2bb 注线（owed=2，如果英雄当前 bet=0）

# 注意：这里假设你的牌编码是 rank-major： card_id = rank*4 + suit，
# rank: 0..12（2..A），suit: 0..3。若你的实现是 suit-major，请把 rank_of()/make_card() 改成 %13 / s*13+rank。
def rank_of(card_id: int) -> int:
    return card_id // 4   # 若 suit-major，请改为: return card_id % 13

def make_card(rank_eng: int, suit: int) -> int:
    return rank_eng * 4 + suit   # 若 suit-major：return suit * 13 + rank_eng

# 我们把“网格行列”map为发动机rank：行/列0对应A → 发动机 rank=12；行/列12对应2 → 发动机 rank=0
def grid_idx_to_engine_rank(idx: int) -> int:
    # idx: 0..12 对应 A..2
    return idx-12

def is_suited_cell(i: int, j: int) -> bool:
    # 业界习惯：上三角多做 (offsuit)，下三角多做 (suited)；这里定义：
    #   i>j: suited（行更强在左下半区）
    #   i<j: offsuit
    #   i==j: pocket pair
    return i < j

def sample_combo_cards(i: int, j: int, rng: np.random.Generator) -> tuple[int, int]:
    """从网格坐标(i,j)随机采样一个具体两张牌（用 card_id 表示）。"""
    r1 = grid_idx_to_engine_rank(i)
    r2 = grid_idx_to_engine_rank(j)
    if i == j:
        # 口袋对子：同等级、不同花色
        s1 = int(rng.integers(0, 4))
        s2 = (s1 + int(rng.integers(1, 4))) % 4
        return make_card(r1, s1), make_card(r2, s2)
    elif is_suited_cell(i, j):
        # 同花：两牌 suits 相同
        s = int(rng.integers(0, 4))
        return make_card(r1, s), make_card(r2, s)
    else:
        # 不同花：两牌 suits 不同
        s1 = int(rng.integers(0, 4))
        s2 = (s1 + int(rng.integers(1, 4))) % 4
        return make_card(r1, s1), make_card(r2, s2)

def build_algo(ckpt_dir: Path):
    algo = Algorithm.from_checkpoint(ckpt_dir.as_posix())
    return algo


def build_base_obs(env) -> dict:
    """从 env.reset() 拿一个合法模板，再定值到“翻前 vs 2bb”的情景。"""
    base_obs, _ = env.reset(seed=2024)
    obs = dict(base_obs)
    obs["pot"] = np.int32(POT_FACING_2BB)
    obs["current_bet"] = np.int32(CURRENT_BET)
    obs["board"] = np.full(5, SENTINEL, dtype=np.int32)
    obs["board_len"] = 0
    obs["hero_bet"] = np.int32(0)
    obs["hero_cont"] = np.int32(0)
    obs["hero_stack"] = np.int32(DEFAULT_STACK)
    obs["history"] = np.full_like(obs["history"], SENTINEL)
    return obs

def classify_action(a) -> str:
    """将动作映射到 'fold'/'call'/'raise' 三类（遇到 atype==1 在欠款时等同于 'call'）。"""
    # 预期动作是 Dict({'atype': int, 'r': float})
    atype = None
    if isinstance(a, dict):
        atype = int(a.get("atype", 1))
    else:
        # 兜底：其他结构时直接当作“跟注”
        return "call"
    if atype == 0:
        return "fold"
    elif atype in (1, 2):
        return "call"
    elif atype == 3:
        return "raise"
    return "call"


def evaluate_grid(ckpt_dir: Path, samples_per_combo: int, combos_per_cell: int, seed: int = 7):
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    algo = build_algo(ckpt_dir)

    # 用环境只为拿 observation_space & 构造模板
    env = NLHEParamEnv(seed=seed, sb=SB, bb=BB, start_stack=DEFAULT_STACK)
    obs_space = env.observation_space
    obs_tmpl = build_base_obs(env)

    # 拿到 RLModule 与设备
    module = algo.get_module()
    device = next(module.parameters()).device if hasattr(module, "parameters") else "cpu"

    # 一个与 FlattenObservations 对齐的手工“扁平化”函数
    # 规则：Dict字段按下面固定顺序拼接；
    # - Discrete 用 one-hot（这里只对 board_len 这么做）
    # - Box/数组直接 .reshape(-1) 接成 1D
    def flatten_obs_like_connector(o: dict) -> np.ndarray:
        parts = []
        parts.append(np.asarray(o["board"], dtype=np.float32).reshape(-1))          # 5
        bl = int(o["board_len"])
        bl_onehot = np.zeros(6, dtype=np.float32); bl_onehot[bl] = 1.0              # 6
        parts.append(bl_onehot)
        for k in ["current_bet", "hero_bet", "hero_cont", "hero_stack", "pot"]:     # 5 × 1
            parts.append(np.array([o[k]], dtype=np.float32))
        parts.append(np.asarray(o["hero_hole"], dtype=np.float32).reshape(-1))      # 2
        parts.append(np.asarray(o["history"], dtype=np.float32).reshape(-1))        # 64*4
        return np.concatenate(parts, axis=0)                                        # shape = [D]

    rng = np.random.default_rng(seed)
    n = len(RANKS)
    fold_grid = np.zeros((n, n), dtype=np.float64)
    call_grid = np.zeros((n, n), dtype=np.float64)
    raise_grid = np.zeros((n, n), dtype=np.float64)

    for i in tqdm.tqdm(range(n), desc="Evaluating grid"):
        for j in tqdm.tqdm(range(n), desc=f"Row {i+1}/{n}", leave=False):
            counts = {"fold": 0, "call": 0, "raise": 0}
            total = 0
            for _ in tqdm.tqdm(range(combos_per_cell), desc=f"Cell {i+1},{j+1}", leave=False):
                c1, c2 = sample_combo_cards(i, j, rng)
                obs = dict(obs_tmpl)
                obs["hero_hole"] = np.array([c1, c2], dtype=np.int32)

                flat = flatten_obs_like_connector(obs)              # np.float32 [D]
                obs_batch = torch.from_numpy(flat[None, ...]).to(device)  # [B=1, D]

                for _ in range(samples_per_combo):
                    with torch.no_grad():
                        out = module.forward_inference({"obs": obs_batch})
                        dist_cls = module.get_inference_action_dist_cls()
                        logits = out["action_dist_inputs"]

                        # 兼容：有些版本提供 classmethod from_logits，有些直接 __init__(logits)
                        if hasattr(dist_cls, "from_logits"):
                            dist = dist_cls.from_logits(logits)
                        else:
                            dist = dist_cls(logits)

                        act_t = dist.sample()                       # Tensor 或 Dict[T]
                    # 转成 python/numpy
                    act = act_t.cpu().numpy() if hasattr(act_t, "cpu") else act_t

                    # 你的动作是 Dict({'atype':int, 'r':float})；若是张量/ndarray，就兜底当作 call
                    tag = classify_action(act if isinstance(act, dict) else {})
                    counts[tag] += 1
                    total += 1

            if total > 0:
                fold_grid[i, j] = counts["fold"] / total
                call_grid[i, j] = counts["call"] / total
                raise_grid[i, j] = counts["raise"] / total

    return fold_grid, call_grid, raise_grid

def save_csv_png(fold_g, call_g, raise_g, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    header = "," + ",".join(list(RANKS))
    for name, g in [("fold", fold_g), ("call", call_g), ("raise", raise_g)]:
        path = out_dir / f"{name}_grid.csv"
        with path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            for i, r in enumerate(RANKS):
                row = [r] + [f"{g[i,j]:.4f}" for j in range(len(RANKS))]
                f.write(",".join(row) + "\n")
        print(f"saved {path}")

    # PNG（matplotlib，单图单色系，不依赖 seaborn）
    def plot_grid(g, title, path):
        fig = plt.figure(figsize=(6.2, 5.8))
        ax = plt.gca()
        im = ax.imshow(g, origin="upper", aspect="equal")
        ax.set_xticks(range(len(RANKS[::-1])))  # 反转 x 轴
        ax.set_xticklabels(list(RANKS[::-1]))
        ax.set_yticks(range(len(RANKS)))
        ax.set_yticklabels(list(RANKS))
        ax.set_xlabel("第二张牌（列）")
        ax.set_ylabel("第一张牌（行）")
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("概率", rotation=90)
        plt.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"saved {path}")

    plot_grid(fold_g,  "FOLD Probability [pre-flop vs 2bb]", out_dir / "fold_grid.png")
    plot_grid(call_g,  "CALL Probability [pre-flop vs 2bb]", out_dir / "call_grid.png")
    plot_grid(raise_g, "RAISE Probability [pre-flop vs 2bb]", out_dir / "raise_grid.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="RLlib checkpoint 目录（包含 env_runner/learner_group 的那个目录）")
    parser.add_argument("--samples-per-combo", type=int, default=64,
                        help="每个具体两张牌组合采样多少次动作（explore=True）")
    parser.add_argument("--combos-per-cell", type=int, default=24,
                        help="每个网格随机抽多少个具体两张牌组合")
    parser.add_argument("--out", type=str, default="./eval_out", help="输出目录")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt).resolve()
    assert ckpt_dir.exists(), f"checkpoint 路径不存在：{ckpt_dir}"

    fold_g, call_g, raise_g = evaluate_grid(
        ckpt_dir=ckpt_dir,
        samples_per_combo=args.samples_per_combo,
        combos_per_cell=args.combos_per_cell,
    )
    save_csv_png(fold_g, call_g, raise_g, Path(args.out))

if __name__ == "__main__":
    main()
