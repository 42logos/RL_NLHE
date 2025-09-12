# eval.py — updated to current settings (RLModule, custom flattener, env registration)

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

import ray
import gymnasium as gym
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns

from nlhe.envs.param_env import NLHEParamEnv
from nlhe.train.flattener import NLHEFlattener, FlattenObsWrapper

# ------------------ 常量 ------------------
RANKS = "AKQJT98765432"       # 13 个等级（A 在左上角）
SENTINEL = -1                 # 未揭示牌占位符
DEFAULT_STACK = 100
SB, BB = 1, 2                 # 小盲/大盲
POT_FACING_2BB = SB + BB      # 翻前未加注 pot≈3（这里简化为3）
CURRENT_BET = 2               # 面对 2bb 注线（英雄当前 bet=0 时欠款=2）

# ------------------ 环境注册（用于 from_checkpoint 重建） ------------------
def _ensure_env_registered(env_id: str = "nlhe_flat") -> None:
    # Gym registry check (Gymnasium >=0.29)
    try:
        if env_id in gym.envs.registry:
            return
    except Exception:
        pass

    def make_env(env_config):
        # env_config is a dict-like EnvContext
        hero_seat   = env_config.get("hero_seat", 0)
        bb          = env_config.get("bb", 2)
        sb          = env_config.get("sb", bb / 2)
        seed        = env_config.get("seed", 0)
        start_stack = env_config.get("start_stack", 100)
        history_len = env_config.get("history_len", 64)

        base = NLHEParamEnv(
            hero_seat=hero_seat,
            bb=bb,
            sb=sb,
            seed=seed,
            start_stack=start_stack,
            history_len=history_len,
        )
        return FlattenObsWrapper(base, history_len=history_len)

    tune.register_env(env_id, lambda cfg: make_env(cfg))

# ------------------ 扑克网格工具 ------------------
def rank_of(card_id: int) -> int:
    return card_id // 4  # rank-major (2..A => 0..12)

def make_card(rank_eng: int, suit: int) -> int:
    return rank_eng * 4 + suit  # rank-major

def grid_idx_to_engine_rank(idx: int) -> int:
    # 网格 0..12 映射到引擎 rank: 12..0
    return idx - 12

def is_suited_cell(i: int, j: int) -> bool:
    # i<j: offsuit, i>j: suited, i==j: pair
    return i < j

def sample_combo_cards(i: int, j: int, rng: np.random.Generator) -> tuple[int, int]:
    r1 = grid_idx_to_engine_rank(i)
    r2 = grid_idx_to_engine_rank(j)
    if i == j:
        s1 = int(rng.integers(0, 4))
        s2 = (s1 + int(rng.integers(1, 4))) % 4
        return make_card(r1, s1), make_card(r2, s2)
    elif is_suited_cell(i, j):
        s = int(rng.integers(0, 4))
        return make_card(r1, s), make_card(r2, s)
    else:
        s1 = int(rng.integers(0, 4))
        s2 = (s1 + int(rng.integers(1, 4))) % 4
        return make_card(r1, s1), make_card(r2, s2)

# ------------------ RLlib 构建 ------------------
def build_algo(ckpt_dir: Path) -> Algorithm:
    _ensure_env_registered("nlhe_flat")  # must be before from_checkpoint
    return Algorithm.from_checkpoint(ckpt_dir.as_posix())

# 用环境只为拿一个“模板 obs”，然后手动设置翻前场景
def build_base_obs() -> dict:
    env = NLHEParamEnv(seed=2024, sb=SB, bb=BB, start_stack=DEFAULT_STACK)
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
    # 预期 Dict({'atype': int, 'r': float}); 其它结构兜底为 'call'
    if isinstance(a, dict):
        atype = int(a.get("atype", 1))
        if atype == 0:
            return "fold"
        elif atype in (1, 2):
            return "call"
        elif atype == 3:
            return "raise"
    return "call"

# ------------------ 主评估逻辑 ------------------
def evaluate_grid(ckpt_dir: Path, samples_per_combo: int, combos_per_cell: int, seed: int = 7):
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    algo = build_algo(ckpt_dir)

    # RLModule + device
    module = algo.get_module() or algo.get_module("default_policy")
    device = next(module.parameters()).device if hasattr(module, "parameters") else torch.device("cpu")

    # 统一扁平器（与训练一致）
    flattener = NLHEFlattener(history_len=64)
    obs_tmpl = build_base_obs()

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

                # 扁平化 + 构造 batch（B*T=1, F）+ RNN 初始状态 + SEQ_LENS=1
                flat = flattener.transform(obs).astype(np.float32)
                obs_batch = torch.as_tensor(flat, device=device).unsqueeze(0)  # [1, F]

                try:
                    init_state = module.get_initial_state()
                except Exception:
                    init_state = []
                state_in = []
                for x in init_state:
                    t = torch.as_tensor(x, device=device)
                    if t.ndim == 1:
                        t = t.unsqueeze(0)  # [1, H]
                    state_in.append(t)

                batch = {
                    Columns.OBS: obs_batch,
                    Columns.SEQ_LENS: torch.tensor([1], dtype=torch.int32, device=device),
                }
                if state_in:
                    batch[Columns.STATE_IN] = state_in

                for _k in range(samples_per_combo):
                    with torch.no_grad():
                        out = module.forward_inference(batch)
                        if Columns.ACTIONS in out:
                            act_t = out[Columns.ACTIONS]
                        else:
                            dist_cls = module.get_inference_action_dist_cls()
                            logits = out[Columns.ACTION_DIST_INPUTS]
                            if hasattr(dist_cls, "from_logits"):
                                dist = dist_cls.from_logits(logits)
                            else:
                                dist = dist_cls(logits)
                            # 采样（随机，非确定）
                            act_t = dist.sample()

                    act = act_t.cpu().numpy() if hasattr(act_t, "cpu") else act_t
                    tag = classify_action(act if isinstance(act, dict) else {})
                    counts[tag] += 1
                    total += 1

            if total > 0:
                fold_grid[i, j] = counts["fold"] / total
                call_grid[i, j] = counts["call"] / total
                raise_grid[i, j] = counts["raise"] / total

    return fold_grid, call_grid, raise_grid

# ------------------ 保存输出 ------------------
def save_csv_png(fold_g, call_g, raise_g, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    header = "," + ",".join(list(RANKS))

    for name, g in [("fold", fold_g), ("call", call_g), ("raise", raise_g)]:
        path = out_dir / f"{name}_grid.csv"
        with path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            for i, r in enumerate(RANKS):
                row = [r] + [f"{g[i, j]:.4f}" for j in range(len(RANKS))]
                f.write(",".join(row) + "\n")
        print(f"saved {path}")

    def plot_grid(g, title, path):
        fig = plt.figure(figsize=(6.2, 5.8))
        ax = plt.gca()
        im = ax.imshow(g, origin="upper", aspect="equal")
        ax.set_xticks(range(len(RANKS[::-1])))
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

# ------------------ CLI ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="RLlib checkpoint 目录（包含 env_runner/learner_group 的目录）")
    parser.add_argument("--samples-per-combo", type=int, default=64,
                        help="每个具体两张牌组合采样次数（explore 模式）")
    parser.add_argument("--combos-per-cell", type=int, default=24,
                        help="每个网格随机抽取的具体两张牌组合个数")
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
