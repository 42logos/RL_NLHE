from ray.rllib.algorithms.ppo import PPOConfig

def restore_algorithm(ckpt):
    # 1) Load the original config saved inside the checkpoint
    config = PPOConfig.from_checkpoint(ckpt)            # <- works for all AlgoConfig subclasses

    # 2) Change training EnvRunner parallelism and vectorization
    config = (config
              .env_runners(
                  num_env_runners=1,                # number of EnvRunner actors (processes)
              )
              .resources(num_gpus=0))               # e.g., override resources if moving to CPU-only

    # 3) Build a new Algorithm with the updated config
    algo = config.build()
    # 4) Restore model/optimizer/etc. from checkpoint
    algo.restore(ckpt)
    
    return algo
