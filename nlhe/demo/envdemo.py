from ..envs.param_env import NLHEParamEnv



if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--human_seat", type=int, default=0)
    args = ap.parse_args()

    env = NLHEParamEnv(
        seed=args.seed,
        hero_seat=args.human_seat
    )
    