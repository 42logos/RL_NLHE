from ray.rllib.algorithms.algorithm import Algorithm

def restore_algorithm(
    ckpt: str,
    num_env_runners: int = 1,
    num_envs_per_env_runner: int = 1,
    remote_worker_envs: bool = False,
    eval_num_env_runners: int = 1,
    cpu_only: bool = False,
):
    # 1) Load algo to access its (saved) config
    algo = Algorithm.from_checkpoint(ckpt)

    # If no overrides requested, use as-is
    if all(v is None for v in [num_env_runners, num_envs_per_env_runner, remote_worker_envs, eval_num_env_runners]) and not cpu_only:
        return algo

    # 2) Copy and mutate config
    cfg = algo.config.copy()

    # Optional resource override (e.g., move GPU->CPU)
    if cpu_only:
        cfg = cfg.resources(num_gpus=0)

    # Training env-runner overrides
    env_opts = {}
    if num_env_runners is not None:
        env_opts["num_env_runners"] = num_env_runners
    if num_envs_per_env_runner is not None:
        env_opts["num_envs_per_env_runner"] = num_envs_per_env_runner
    if remote_worker_envs is not None:
        env_opts["remote_worker_envs"] = remote_worker_envs
    if env_opts:
        cfg = cfg.env_runners(**env_opts)  # new API stack knobs

    # Evaluation env-runner override
    if eval_num_env_runners is not None:
        cfg = cfg.evaluation(evaluation_num_env_runners=eval_num_env_runners)

    # 3) Build fresh algo and restore the saved state into it
    new_algo = cfg.build()
    if hasattr(new_algo, "restore_from_path"):
        new_algo.restore_from_path(ckpt)
    else:  # older alias, just in case
        new_algo.restore(ckpt)

    # Clean up the temp algo created by from_checkpoint
    try:
        algo.stop()
    except Exception:
        pass

    return new_algo
