from __future__ import annotations
import random
from typing import List, Optional
from ..core.rs_engine import NLHEngine
from ..core.types import GameState
from ..agents.tamed_random import TamedRandomAgent
from ..agents.human_cli import HumanAgent

# Simple renderer (text)
RSTR = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
SUIT = ['♣','♦','♥','♠']
def card_str(rank_of, suit_of, c:int)->str:
    r = rank_of(c); s = suit_of(c); rs = str(r) if r<=10 else RSTR[r]; return f"{rs}{SUIT[s]}"
def cards_str(rank_of, suit_of, cards): return ' '.join(card_str(rank_of, suit_of, c) for c in cards)

def render_table(env: NLHEngine, s: GameState, hero: int, reveal_all: bool=False) -> None:
    from ..core.cards import rank_of, suit_of
    print("\n=== NLHE 6-Max ===")
    print(f"Button: {s.button} | Round: {s.round_label} | Pot: {s.pot} | CurrentBet: {s.current_bet}")
    if s.board:
        street = {3:'Flop',4:'Turn',5:'River'}.get(len(s.board), 'Board')
        print(f"Board ({street}): {cards_str(rank_of, suit_of, s.board)}")
    else:
        print("Board: (preflop)")
    print("Seats:")
    for i, p in enumerate(s.players):
        if reveal_all and p.hole:
            hole = f"{card_str(rank_of, suit_of, p.hole[0])} {card_str(rank_of, suit_of, p.hole[1])}"
        else:
            hole = (f"{card_str(rank_of, suit_of, p.hole[0])} {card_str(rank_of, suit_of, p.hole[1])}" if i==hero and p.hole else '?? ??')
        tag = ' (BTN)' if i==s.button else ''
        print(f"  Seat {i}{tag}: status={p.status:6} stack={p.stack:3} bet={p.bet:3} cont={p.cont:3} hole={hole}")
    if s.next_to_act is not None:
        print(f"Next to act: Seat {s.next_to_act}")

def run_hand_cli(seed: int = 42, button: int = 0, human_seat: int = 0) -> None:
    rng = random.Random(seed)
    env = NLHEngine(sb=1, bb=2, start_stack=100, rng=rng)
    s = env.reset_hand(button=button)
    agents: List = [TamedRandomAgent(rng) for _ in range(env.N)]
    agents[human_seat] = HumanAgent()

    done = False; rewards = None
    while not done:
        render_table(env, s, hero=human_seat)
        if s.next_to_act is None:
            done, rewards = env.advance_round_if_needed(s)
            if done: break
            continue
        seat = s.next_to_act
        agent = agents[seat]
        a = agent.act(env, s, seat)
        s, done, rewards, _ = env.step(s, a)

    print("\n=== Hand Result ===")
    render_table(env, s, hero=human_seat, reveal_all=True)
    print("Rewards (win - cont):", rewards)
    print("Sum:", sum(rewards))

if __name__ == "__main__":
    human_seat = 0
    print(f"Interactive NLHE demo — you are Seat {human_seat}. Commands: f/k/c, 'r X', 'allin'.")
    run_hand_cli(seed=42, button=0, human_seat=human_seat)
