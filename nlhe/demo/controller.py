from __future__ import annotations

import random
from typing import List, Optional

from PyQt6 import QtCore

from ..agents.tamed_random import TamedRandomAgent
from ..agents.ckpt_agent import CKPTAgent
from ..core.engine import NLHEngine
from ..core.types import Action, GameState


class GameController(QtCore.QObject):
    """Manage NLHEngine state and autonomous agents.

    The controller encapsulates game progression so that interfaces such as
    ``NLHEGui`` only need to handle rendering and user input.  It exposes a
    handful of Qt signals to notify listeners about state changes, action logs
    and when a hand has completed.
    """

    state_changed = QtCore.pyqtSignal(GameState)
    hand_finished = QtCore.pyqtSignal(list)
    action_logged = QtCore.pyqtSignal(int, Action)

    def __init__(self, hero_seat: int = 0, seed: int = 42,
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.hero_seat = hero_seat
        self.seed_val = seed
        self.rng = random.Random(seed)
        self.engine = NLHEngine(sb=1, bb=2, start_stack=100, rng=self.rng)
        self.agents: List[TamedRandomAgent | CKPTAgent | None] = [
            TamedRandomAgent(self.rng) for _ in range(self.engine.N)
        ]
        self.agents[hero_seat] = None  # human
        self.state: GameState = self.engine.reset_hand(button=0)
        self.state_changed.emit(self.state)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.play_loop)
        self.timer.start()

    # ------------------------------------------------------------------
    def play_loop(self) -> None:
        """Advance the game by allowing the next agent to act."""
        if self.state.next_to_act is None:
            done, rewards = self.engine.advance_round_if_needed(self.state)
            if done:
                self._end_hand(rewards)
                return
            self.state_changed.emit(self.state)
            return

        seat = self.state.next_to_act
        if seat == self.hero_seat:
            return  # wait for user

        agent = self.agents[seat]
        assert agent is not None
        action = agent.act(self.engine, self.state, seat)
        self.action_logged.emit(seat, action)
        self.state, done, rewards, _ = self.engine.step(self.state, action)
        if done:
            self._end_hand(rewards)
            return
        if self.state.next_to_act is None:
            done, rewards = self.engine.advance_round_if_needed(self.state)
            if done:
                self._end_hand(rewards)
                return
        self.state_changed.emit(self.state)

    # ------------------------------------------------------------------
    def submit_action(self, action: Action) -> None:
        """Apply a user action for the hero seat."""
        if self.state.next_to_act != self.hero_seat:
            return
        self.action_logged.emit(self.hero_seat, action)
        self.state, done, rewards, _ = self.engine.step(self.state, action)
        if done:
            self._end_hand(rewards)
            return
        if self.state.next_to_act is None:
            done, rewards = self.engine.advance_round_if_needed(self.state)
            if done:
                self._end_hand(rewards)
                return
        self.state_changed.emit(self.state)

    # ------------------------------------------------------------------
    def start_next_hand(self, seed: Optional[int] = None) -> None:
        """Reset the engine for a new hand and restart the loop."""
        if seed is not None:
            self.seed_val = seed
        self.rng.seed(self.seed_val)
        button = (self.state.button + 1) % self.engine.N
        self.state = self.engine.reset_hand(button=button)
        self.state_changed.emit(self.state)
        self.timer.start()

    # ------------------------------------------------------------------
    def _end_hand(self, rewards: List[int]) -> None:
        self.timer.stop()
        # derive a fresh seed for the next hand so card order changes
        new_seed = random.Random(self.seed_val).randrange(1 << 30)
        self.seed_val = new_seed
        self.hand_finished.emit(rewards)
