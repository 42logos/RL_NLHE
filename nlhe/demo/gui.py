from __future__ import annotations

import random
from typing import List

from PyQt6 import QtWidgets, QtCore

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


class NLHEGui(QtWidgets.QMainWindow):
    """PyQt6 GUI to play a single NLHE hand against random agents."""

    def __init__(self, hero_seat: int = 0, seed: int = 42) -> None:
        super().__init__()
        self.setWindowTitle("NLHE 6-Max GUI")
        self.hero_seat = hero_seat
        self.rng = random.Random(seed)
        self.engine = NLHEngine(sb=1, bb=2, start_stack=100, rng=self.rng)
        self.agents: List = [TamedRandomAgent(self.rng) for _ in range(self.engine.N)]
        self.agents[hero_seat] = None  # human
        self.state: GameState = self.engine.reset_hand(button=0)

        self._create_widgets()
        self._update_view()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._play_loop)
        self.timer.start(500)

    def _create_widgets(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.board_label = QtWidgets.QLabel("Board: ")
        layout.addWidget(self.board_label)

        self.player_labels: List[QtWidgets.QLabel] = []
        for _ in range(self.engine.N):
            lbl = QtWidgets.QLabel()
            layout.addWidget(lbl)
            self.player_labels.append(lbl)

        btn_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_layout)
        self.action_buttons = {}
        for name in ["FOLD", "CHECK", "CALL", "RAISE"]:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _, n=name: self._on_action(n))
            btn_layout.addWidget(btn)
            self.action_buttons[name] = btn
        self.raise_entry = QtWidgets.QLineEdit()
        self.raise_entry.setFixedWidth(60)
        btn_layout.addWidget(self.raise_entry)

        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

        self.setCentralWidget(central)

    def _update_view(self) -> None:
        board_text = f"Board: {cards_str(self.state.board)}" if self.state.board else "Board: (preflop)"
        self.board_label.setText(board_text)

        for i, lbl in enumerate(self.player_labels):
            p = self.state.players[i]
            if i == self.hero_seat and p.hole:
                hole = cards_str(list(p.hole))
            else:
                hole = "?? ??"
            text = (
                f"Seat {i} | stack={p.stack:3} bet={p.bet:3} cont={p.cont:3} "
                f"status={p.status:6} hole={hole}"
            )
            lbl.setText(text)

        if self.state.next_to_act is not None:
            self.status_label.setText(f"Next to act: Seat {self.state.next_to_act}")
        else:
            self.status_label.setText("Waiting for round advance...")

        info = self.engine.legal_actions(self.state)
        allowed = {a.kind for a in info.actions}
        self.action_buttons["FOLD"].setEnabled(ActionType.FOLD in allowed)
        self.action_buttons["CHECK"].setEnabled(ActionType.CHECK in allowed)
        self.action_buttons["CALL"].setEnabled(ActionType.CALL in allowed)
        raise_allowed = ActionType.RAISE_TO in allowed
        self.action_buttons["RAISE"].setEnabled(raise_allowed)
        self.raise_entry.setEnabled(raise_allowed)
        self.min_raise_to = getattr(info, "min_raise_to", None)
        self.max_raise_to = getattr(info, "max_raise_to", None)

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
                amt = int(self.raise_entry.text())
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "Invalid", "Enter raise amount")
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

    def _end_hand(self, rewards: List[int]) -> None:
        for i, lbl in enumerate(self.player_labels):
            p = self.state.players[i]
            hole = cards_str(list(p.hole)) if p.hole else "?? ??"
            lbl.setText(lbl.text() + f" | hole={hole}")
        msg = "\n".join(f"Seat {i}: {r}" for i, r in enumerate(rewards))
        QtWidgets.QMessageBox.information(self, "Hand complete", msg)
        for btn in self.action_buttons.values():
            btn.setEnabled(False)
        self.raise_entry.setEnabled(False)
        self.status_label.setText("Hand complete")
        self.timer.stop()


def main() -> None:
    import sys

    app = QtWidgets.QApplication(sys.argv)
    gui = NLHEGui(hero_seat=0, seed=42)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
