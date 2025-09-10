"""PyQt6-based demo GUI for playing a single NLHE hand.

This modernised interface lets a human play one 6-max No-Limit Texas
Hold'em hand against basic random agents.  It demonstrates interaction with
the ``NLHEngine`` API while providing a minimal but cleaner look compared to
the previous Tkinter demo.
"""

from __future__ import annotations

import random
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets

from ..agents.tamed_random import TamedRandomAgent
from ..core.cards import rank_of, suit_of
from ..core.engine import NLHEngine
from ..core.types import Action, ActionType, GameState


RSTR = {11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT = ["♣", "♦", "♥", "♠"]


def card_str(c: int) -> str:
    r = rank_of(c)
    s = suit_of(c)
    rs = str(r) if r <= 10 else RSTR[r]
    return f"{rs}{SUIT[s]}"


def cards_str(cards: List[int]) -> str:
    return " ".join(card_str(c) for c in cards)


class NLHEGui(QtWidgets.QMainWindow):
    """Simple PyQt6 GUI to play a single NLHE hand against random agents."""

    def __init__(self, hero_seat: int = 0, seed: int = 42) -> None:
        super().__init__()
        self.setWindowTitle("NLHE 6-Max GUI")
        self.hero_seat = hero_seat
        self.rng = random.Random(seed)
        self.engine = NLHEngine(sb=1, bb=2, start_stack=100, rng=self.rng)
        self.agents: List | List[TamedRandomAgent | None] = [
            TamedRandomAgent(self.rng) for _ in range(self.engine.N)
        ]
        self.agents[hero_seat] = None  # human
        self.state: GameState = self.engine.reset_hand(button=0)

        self._create_widgets()
        self._update_view()

        # periodic timer for engine progression
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._play_loop)
        self.timer.start()

    # ----- UI setup -----
    def _create_widgets(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        self.board_label = QtWidgets.QLabel("Board: (preflop)")
        layout.addWidget(self.board_label)

        self.player_labels: List[QtWidgets.QLabel] = []
        for _ in range(self.engine.N):
            lbl = QtWidgets.QLabel("")
            layout.addWidget(lbl)
            self.player_labels.append(lbl)

        btn_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_layout)
        self.action_buttons: dict[str, QtWidgets.QPushButton] = {}
        for name in ["FOLD", "CHECK", "CALL", "RAISE"]:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _, n=name: self._on_action(n))
            btn_layout.addWidget(btn)
            self.action_buttons[name] = btn

        self.raise_edit = QtWidgets.QLineEdit()
        self.raise_edit.setFixedWidth(60)
        self.raise_edit.setPlaceholderText("raise diff")
        btn_layout.addWidget(self.raise_edit)

        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

    # ----- helpers -----
    def _last_action(self, seat: int) -> Optional[tuple[int, int]]:
        """Return the last (aid, amount) for a seat if available."""
        for sid, aid, amt, _ in reversed(self.state.actions_log):
            if sid == seat:
                return aid, amt
        return None

    def _update_view(self) -> None:
        # Board
        if self.state.board:
            self.board_label.setText(f"Board: {cards_str(self.state.board)}")
        else:
            self.board_label.setText("Board: (preflop)")

        # Players
        for i, lbl in enumerate(self.player_labels):
            p = self.state.players[i]
            if i == self.hero_seat and p.hole:
                hole = cards_str(list(p.hole))
            else:
                hole = "?? ??"
            last = self._last_action(i)
            if last is None:
                last_str = ""
            else:
                aid, amt = last
                if aid == 0:
                    last_str = "fold"
                elif aid == 1:
                    last_str = "check"
                elif aid == 2:
                    last_str = "call"
                else:
                    last_str = f"raise to {amt}"
            text = (
                f"Seat {i} | stack={p.stack:3} bet={p.bet:3} cont={p.cont:3}"
                f" status={p.status:6} last={last_str:8} hole={hole}"
            )
            lbl.setText(text)
            if p.status == "folded":
                lbl.setStyleSheet("color: grey")
            elif last_str.startswith("raise"):
                lbl.setStyleSheet("color: green")
            else:
                lbl.setStyleSheet("")

        if self.state.next_to_act is not None:
            seat = self.state.next_to_act
            to_call = self.state.current_bet - self.state.players[seat].bet
            self.status_label.setText(
                f"Next to act: Seat {seat} (to call {to_call})"
            )
        else:
            self.status_label.setText("Waiting for round advance...")

        info = self.engine.legal_actions(self.state)
        allowed = {a.kind for a in info.actions}
        self.action_buttons["FOLD"].setEnabled(ActionType.FOLD in allowed)
        self.action_buttons["CHECK"].setEnabled(ActionType.CHECK in allowed)
        self.action_buttons["CALL"].setEnabled(ActionType.CALL in allowed)
        raise_allowed = ActionType.RAISE_TO in allowed
        self.action_buttons["RAISE"].setEnabled(raise_allowed)
        self.raise_edit.setEnabled(raise_allowed)
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
                inc = int(self.raise_edit.text())
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "Invalid", "Enter raise increment")
                return
            target = self.state.current_bet + inc
            if self.min_raise_to is not None and target < self.min_raise_to:
                need = self.min_raise_to - self.state.current_bet
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid",
                    f"Minimum raise diff is {need}",
                )
                return
            if self.max_raise_to is not None and target > self.max_raise_to:
                cap = self.max_raise_to - self.state.current_bet
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid",
                    f"Maximum raise diff is {cap}",
                )
                return
            a = Action(ActionType.RAISE_TO, amount=target)
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

    def _end_hand(self, rewards: List[int]) -> None:
        for i, lbl in enumerate(self.player_labels):
            p = self.state.players[i]
            hole = cards_str(list(p.hole)) if p.hole else "?? ??"
            lbl.setText(lbl.text() + f" | hole={hole}")
        msg = "\n".join(f"Seat {i}: {r}" for i, r in enumerate(rewards))
        QtWidgets.QMessageBox.information(self, "Hand complete", msg)
        for btn in self.action_buttons.values():
            btn.setEnabled(False)
        self.raise_edit.setEnabled(False)
        self.status_label.setText("Hand complete")
        self.timer.stop()


def main() -> None:
    app = QtWidgets.QApplication([])
    gui = NLHEGui(hero_seat=0, seed=42)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()

