import argparse
import random
from nlhe.agents.ckpt_agent import CKPTAgent
from nlhe.core.rs_engine import NLHEngine
from nlhe.core.types import ActionType


def run_demo(ckpt_path: str):
    eng = NLHEngine(rng=random.Random(0))
    state = eng.reset_hand(0)
    agent = CKPTAgent(ckpt_path)
    steps = 0
    while True:
        seat = state.next_to_act
        if seat is None:
            break
        action = agent.act(eng, state, seat)
        print(f"seat {seat} -> {action.kind.name}{'' if action.amount is None else ' '+str(action.amount)}")
        state, done, _, _ = eng.step(state, action)
        if done:
            break
        done, _ = eng.advance_round_if_needed(state)
        if done:
            break
        steps += 1
        if steps > 100:
            print('stopping after 100 steps')
            break
    print('final pot', state.pot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='path to RLlib checkpoint directory')
    args = parser.parse_args()
    run_demo(args.ckpt)
