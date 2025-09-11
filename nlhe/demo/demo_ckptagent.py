from ..envs.param_env import NLHEParamEnv
from ..agents.ckpt_agent import CKPTAgent
from ..agents.tamed_random import TamedRandomAgent
from ..core.types import GameState
from typing import Dict, Any



def run_demo_ckpt_agent(checkpoint_path: str, seed: int = 42, button: int = 0, hero_seat: int = 0) -> None:
    import random
    from typing import List
    from ..core.engine import NLHEngine
    from ..core.types import GameState
    from gymnasium import spaces

    rng = random.Random(seed)
    env = NLHEngine(sb=1, bb=2, start_stack=100, rng=rng)
    s = env.reset_hand(button=button)

    # Create agents
    agents: List = [TamedRandomAgent(rng) for _ in range(env.N)]
    from ..agents.human_cli import HumanAgent
    agents[hero_seat] = HumanAgent()
    # env.N mod (hero_seat +1) is for ckpt agent to be next to act after hero
    ckpt_seat = (hero_seat + 1) % env.N
    agents[ckpt_seat] = CKPTAgent(checkpoint_path)
    print(f"Using CKPTAgent at seat {ckpt_seat}")
    print(f"Using HumanAgent at seat {hero_seat}")

    done = False
    rewards = None
    while not done:
        # Render the table
        from .cli import render_table
        render_table(env, s, hero=hero_seat)
        if s.next_to_act is None:
            done, rewards = env.advance_round_if_needed(s)
            if done:
                break
            continue
        seat = s.next_to_act
        agent = agents[seat]
        a = agent.act(env, s, seat)
        s, done, rewards, _ = env.step(s, a)

    print("\n=== Hand Result ===")
    render_table(env, s, hero=hero_seat, reveal_all=True)
    print("Rewards (win - cont):", rewards)
    print("Sum:", sum(rewards))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demo CKPTAgent in a CLI NLHE game")
    parser.add_argument("--checkpoint", type=str, help="Path to RLlib checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--button", type=int, default=0, help="Button seat (0-5)")
    parser.add_argument("--hero_seat", type=int, default=0, help="Human player seat (0-5)")
    args = parser.parse_args()
    run_demo_ckpt_agent(args.checkpoint, seed=args.seed, button=args.button, hero_seat=args.hero_seat)