from __future__ import annotations
import random
import tkinter as tk
from tkinter import messagebox
from typing import List

from ..core.engine import NLHEngine
from ..core.types import Action, ActionType, GameState
from ..agents.tamed_random import TamedRandomAgent
from ..core.cards import rank_of, suit_of

RSTR = {11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT = ["♣", "♦", "♥", "♠"]

def card_str(c: int) -> str:
    r = rank_of(c)
    s = suit_of(c)
    rs = str(r) if r <= 10 else RSTR[r]
    return f"{rs}{SUIT[s]}"

def cards_str(cards: List[int]) -> str:
    return " ".join(card_str(c) for c in cards)


class NLHEGui(tk.Tk):
    """Simple Tkinter GUI to play a single NLHE hand against random agents."""

    def __init__(self, hero_seat: int = 0, seed: int = 42) -> None:
        super().__init__()
        self.title("NLHE 6-Max GUI")
        self.hero_seat = hero_seat
        self.rng = random.Random(seed)
        self.engine = NLHEngine(sb=1, bb=2, start_stack=100, rng=self.rng)
        self.agents: List = [TamedRandomAgent(self.rng) for _ in range(self.engine.N)]
        self.agents[hero_seat] = None  # human
        self.state: GameState = self.engine.reset_hand(button=0)

        self._create_widgets()
        self._update_view()
        self.after(500, self._play_loop)

    # ----- UI setup -----
    def _create_widgets(self) -> None:
        self.board_label = tk.Label(self, text="Board: ")
        self.board_label.pack(pady=5)

        self.player_labels: List[tk.Label] = []
        for i in range(self.engine.N):
            lbl = tk.Label(self, text="")
            lbl.pack(anchor="w")
            self.player_labels.append(lbl)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=5)
        self.action_buttons = {}
        for name in ["FOLD", "CHECK", "CALL", "RAISE"]:
            btn = tk.Button(btn_frame, text=name, command=lambda n=name: self._on_action(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.action_buttons[name] = btn
        self.raise_entry = tk.Entry(btn_frame, width=6)
        self.raise_entry.pack(side=tk.LEFT, padx=2)

        self.status_label = tk.Label(self, text="")
        self.status_label.pack(pady=5)

    # ----- helpers -----
    def _update_view(self) -> None:
        # Board
        if self.state.board:
            self.board_label.config(text=f"Board: {cards_str(self.state.board)}")
        else:
            self.board_label.config(text="Board: (preflop)")

        # Players
        for i, lbl in enumerate(self.player_labels):
            p = self.state.players[i]
            if i == self.hero_seat and p.hole:
                hole = cards_str(list(p.hole))
            else:
                hole = "?? ??"
            text = (
                f"Seat {i} | stack={p.stack:3} bet={p.bet:3} cont={p.cont:3}"
                f" status={p.status:6} hole={hole}"
            )
            lbl.config(text=text)

        if self.state.next_to_act is not None:
            self.status_label.config(text=f"Next to act: Seat {self.state.next_to_act}")
        else:
            self.status_label.config(text="Waiting for round advance...")

        # Action buttons based on legal actions
        info = self.engine.legal_actions(self.state)
        allowed = {a.kind for a in info.actions}
        self.action_buttons["FOLD"].config(state=tk.NORMAL if ActionType.FOLD in allowed else tk.DISABLED)
        self.action_buttons["CHECK"].config(state=tk.NORMAL if ActionType.CHECK in allowed else tk.DISABLED)
        self.action_buttons["CALL"].config(state=tk.NORMAL if ActionType.CALL in allowed else tk.DISABLED)
        raise_allowed = ActionType.RAISE_TO in allowed
        self.action_buttons["RAISE"].config(state=tk.NORMAL if raise_allowed else tk.DISABLED)
        self.raise_entry.config(state=tk.NORMAL if raise_allowed else tk.DISABLED)
        self.min_raise_to = getattr(info, "min_raise_to", None)
        self.max_raise_to = getattr(info, "max_raise_to", None)

    # ----- gameplay loop -----
    def _play_loop(self) -> None:
        if self.state.next_to_act is None:
            done, rewards = self.engine.advance_round_if_needed(self.state)
            if done:
                self._end_hand(rewards)
                return
            self._update_view()
            self.after(500, self._play_loop)
            return

        seat = self.state.next_to_act
        if seat == self.hero_seat:
            # wait for user action
            return

        agent = self.agents[seat]
        assert agent is not None
        action = agent.act(self.engine, self.state, seat)
        self.state, done, rewards, _ = self.engine.step(self.state, action)
        if done:
            self._end_hand(rewards)
            return
        if self.state.next_to_act is None:
            done, rewards = self.engine.advance_round_if_needed(self.state)
            if done:
                self._end_hand(rewards)
                return
        self._update_view()
        self.after(500, self._play_loop)

    def _on_action(self, name: str) -> None:
        if self.state.next_to_act != self.hero_seat:
            return
        if name == "FOLD":
            a = Action(ActionType.FOLD)
        elif name == "CHECK":
            a = Action(ActionType.CHECK)
        elif name == "CALL":
            a = Action(ActionType.CALL)
        elif name == "RAISE":
            try:
                amt = int(self.raise_entry.get())
            except ValueError:
                messagebox.showerror("Invalid", "Enter raise amount")
                return
            a = Action(ActionType.RAISE_TO, amount=amt)
        else:
            return

        self.state, done, rewards, _ = self.engine.step(self.state, a)
        if done:
            self._end_hand(rewards)
            return
        if self.state.next_to_act is None:
            done, rewards = self.engine.advance_round_if_needed(self.state)
            if done:
                self._end_hand(rewards)
                return
        self._update_view()
        self.after(500, self._play_loop)

    def _end_hand(self, rewards: List[int]) -> None:
        # reveal all hole cards
        for i, lbl in enumerate(self.player_labels):
            p = self.state.players[i]
            hole = cards_str(list(p.hole)) if p.hole else "?? ??"
            lbl.config(text=lbl.cget("text") + f" | hole={hole}")
        msg = "\n".join(f"Seat {i}: {r}" for i, r in enumerate(rewards))
        messagebox.showinfo("Hand complete", msg)
        for btn in self.action_buttons.values():
            btn.config(state=tk.DISABLED)
        self.raise_entry.config(state=tk.DISABLED)
        self.status_label.config(text="Hand complete")


def main() -> None:
    gui = NLHEGui(hero_seat=0, seed=42)
    gui.mainloop()


if __name__ == "__main__":
    main()
