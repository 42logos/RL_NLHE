import torch

def _force_safe_defaults(opt: torch.optim.Optimizer):
    opt.defaults["foreach"] = False
    opt.defaults["fused"] = False
    opt.defaults["capturable"] = False
    for g in opt.param_groups:
        g["foreach"] = False
        g["fused"] = False
        g["capturable"] = False
        b1, b2 = g.get("betas", (0.9, 0.999))
        if isinstance(b1, torch.Tensor): b1 = float(b1.item())
        if isinstance(b2, torch.Tensor): b2 = float(b2.item())
        g["betas"] = (float(b1), float(b2))

# ---- 1) 补丁 __init__（你已有，保留） ----
_Adam__init = torch.optim.Adam.__init__
def _Adam_init(self, params, **kw):
    kw["foreach"] = False; kw["fused"] = False; kw["capturable"] = False
    b1, b2 = kw.get("betas", (0.9, 0.999))
    if isinstance(b1, torch.Tensor): b1 = float(b1.item())
    if isinstance(b2, torch.Tensor): b2 = float(b2.item())
    kw["betas"] = (float(b1), float(b2))
    _Adam__init(self, params, **kw); _force_safe_defaults(self)
torch.optim.Adam.__init__ = _Adam_init

_AdamW__init = torch.optim.AdamW.__init__
def _AdamW_init(self, params, **kw):
    kw["foreach"] = False; kw["fused"] = False; kw["capturable"] = False
    b1, b2 = kw.get("betas", (0.9, 0.999))
    if isinstance(b1, torch.Tensor): b1 = float(b1.item())
    if isinstance(b2, torch.Tensor): b2 = float(b2.item())
    kw["betas"] = (float(b1), float(b2))
    _AdamW__init(self, params, **kw); _force_safe_defaults(self)
torch.optim.AdamW.__init__ = _AdamW_init

# ---- 2) 关键：补丁 step()，每次 step 前“就地消毒” ----
_Adam_step = torch.optim.Adam.step
def _patched_adam_step(self, *a, **kw):
    _force_safe_defaults(self)     # 每步都修
    return _Adam_step(self, *a, **kw)
torch.optim.Adam.step = _patched_adam_step

_AdamW_step = torch.optim.AdamW.step
def _patched_adamw_step(self, *a, **kw):
    _force_safe_defaults(self)     # 每步都修
    return _AdamW_step(self, *a, **kw)
torch.optim.AdamW.step = _patched_adamw_step

def _sanitize_state_dict(st):
    """把 state_dict 中的 param_groups 修成兼容值。"""
    import torch
    if not isinstance(st, dict) or "param_groups" not in st:
        return st
    for pg in st["param_groups"]:
        # 关掉不兼容的开关
        pg["foreach"] = False
        pg["fused"] = False
        pg["capturable"] = False
        # betas -> 浮点数
        b1, b2 = pg.get("betas", (0.9, 0.999))
        if isinstance(b1, torch.Tensor): b1 = float(b1.item())
        if isinstance(b2, torch.Tensor): b2 = float(b2.item())
        pg["betas"] = (float(b1), float(b2))
    return st

def force_safe_on_algo_optimizers(algo):
    """对 Learner 里的所有 optimizer 做“双保险”：param_groups 改 + state_dict 改并回灌。"""
    import torch
    try:
        learners = algo.learner_group.get_local_learners()
    except Exception:
        return
    for lrnr in learners:
        # 收集 optimizers
        opts = []
        if hasattr(lrnr, "optimizers") and isinstance(lrnr.optimizers, dict):
            opts += list(lrnr.optimizers.values())
        for v in lrnr.__dict__.values():
            if isinstance(v, torch.optim.Optimizer):
                opts.append(v)
        seen = set()
        for opt in opts:
            if id(opt) in seen:
                continue
            seen.add(id(opt))
            # 先直接改当前对象的 defaults/param_groups
            _force_safe_defaults(opt)
            # 再把 state_dict 读出，修正 param_groups，并回灌
            st = opt.state_dict()
            st = _sanitize_state_dict(st)
            opt.load_state_dict(st)
# ------------------------------------------------------------
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.connectors.env_to_module import FlattenObservations
from nlhe.envs.param_env import NLHEParamEnv as NLHEGymParamEnv
import datetime
import pathlib
import os



if __name__ == "__main__":
        
    save_dir = pathlib.Path("./checkpointsPRO").resolve()
    subdirs = list(save_dir.glob("checkpoint*"))
    print("Found checkpoints:", subdirs)
    newest_checkpoint = max(subdirs, key=lambda f: f.stat().st_mtime) # 找到最新的检查点
    print("Loading from:", newest_checkpoint)
    ray.init()
    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=NLHEGymParamEnv,
            env_config={
                "hero_seat": 0,
                "seed": 42,
                "sb": 1,
                "bb": 2,
                "start_stack": 100,
                "history_len": 64,
                "debug_contains_check": False,
            },
        )
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)
        .env_runners(
            num_env_runners=1,  # 本地单进程采样
            # 把 Dict/Discrete 观测自动扁平化(one-hot)，避免 Catalog 报错
            env_to_module_connector=lambda env, spaces, device: [FlattenObservations()],
        )
        .learners(
            num_learners=0,         # 单进程 Learner，避免 Windows 上的 NCCL
            num_gpus_per_learner=1, # 仍然使用 GPU 训练（cuda:0）
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size_per_learner=1024,
            minibatch_size=512,
            num_epochs=10,
        )
    )
    


    
    algo = config.build_algo()  # 新API，避免弃用警告
    try:
        algo.restore(newest_checkpoint.as_posix())
    except Exception as e:
        print("Error loading checkpoint:", e)
    force_safe_on_algo_optimizers(algo)
    for i in range(1435):
        # 关键：每次训练前再“二次保险”消毒一次
        force_safe_on_algo_optimizers(algo)

        result = algo.train()
        print(i, "reward_mean=", result["env_runners"]["episode_return_mean"])

        if i % 10 == 0:
            savepoint = pathlib.Path(os.path.join(
                save_dir, f"checkpoint_{i}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
            )).resolve()
            # 用 URI 最稳
            cp = algo.save(savepoint)
            print("checkpoint saved to", savepoint)

            # 保存后再消毒一次（防止 save 过程中 param_groups 被覆盖）
            force_safe_on_algo_optimizers(algo)