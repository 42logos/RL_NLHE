# import os
# import sys
# from pathlib import Path

# # Add the project root to Python path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.connectors.env_to_module import FlattenObservations

# from nlhe.agents.ckpt_agent import CKPTAgent
# from nlhe.envs.param_env import NLHEParamEnv
# from nlhe.core.engine import NLHEngine
# from gymnasium import spaces
# import numpy as np


# class _BoxBoardLenEnv(NLHEParamEnv):
#     """NLHEParamEnv variant with Box board_len for stable flattening."""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.observation_space["board_len"] = spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32)

#     def _obs(self):
#         obs = super()._obs()
#         obs["board_len"] = np.asarray([obs["board_len"]], dtype=np.int32)
#         return obs

# def test_ckpt_agent_from_checkpoint(tmp_path: Path):
#     config = (
#         PPOConfig()
#         .environment(_BoxBoardLenEnv, env_config={
#             "hero_seat": 1,
#             "bb": 2,
#             "sb": 1,
#             "seed": 0,
#             "start_stack": 100,
#             "history_len": 64,
#         })
#         .training(lr=1e-3, gamma=0.9, train_batch_size_per_learner=32, minibatch_size=16, num_epochs=1)
#         .framework("torch")
#         .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
#         .env_runners(
#             env_to_module_connector=lambda env: [FlattenObservations()],
#             num_env_runners=0
#         )
#         .learners(num_learners=0, num_gpus_per_learner=0)
#     )
#     algo = config.build()
#     algo.train()
#     ckpt_obj = algo.save(str(tmp_path))
#     # Extract the actual checkpoint path from the TrainingResult object
#     if hasattr(ckpt_obj, 'checkpoint') and hasattr(ckpt_obj.checkpoint, 'path'):
#         ckpt_path = ckpt_obj.checkpoint.path  # type: ignore
#     else:
#         raise ValueError(f"Unable to extract checkpoint path from {ckpt_obj}")
#     agent = CKPTAgent(ckpt_path)
#     env = NLHEngine()
#     state = env.reset_hand(button=0)
#     act = agent.act(env, state, seat=1)
#     print(f"Action from checkpoint agent: {act}")
#     assert act is not None

