import os
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig

from nlhe.agents.ckpt_agent import CKPTAgent
from nlhe.envs.param_env import NLHEParamEnv
from nlhe_engine import NLHEngine
from gymnasium import spaces
import numpy as np


class _BoxBoardLenEnv(NLHEParamEnv):
    """NLHEParamEnv variant with Box board_len for stable flattening."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space["board_len"] = spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32)

    def _obs(self):
        obs = super()._obs()
        obs["board_len"] = np.asarray([obs["board_len"]], dtype=np.int32)
        return obs

def test_ckpt_agent_from_checkpoint(tmp_path: Path):
    config = (
        PPOConfig()
        .environment(_BoxBoardLenEnv, env_config={
            "hero_seat": 1,
            "bb": 2,
            "sb": 1,
            "seed": 0,
            "start_stack": 100,
            "history_len": 64,
        })
        .training(lr=1e-3, gamma=0.9, train_batch_size=32, sgd_minibatch_size=16, num_sgd_iter=1)
        .rollouts(num_rollout_workers=0)
        .framework("torch")
    )
    algo = config.build()
    algo.train()
    ckpt_obj = algo.save(str(tmp_path))
    ckpt_path = getattr(getattr(ckpt_obj, "checkpoint", ckpt_obj), "path", ckpt_obj)
    agent = CKPTAgent(ckpt_path)
    env = NLHEngine()
    state = env.reset_hand(button=0)
    act = agent.act(env, state, seat=1)
    assert act is not None
