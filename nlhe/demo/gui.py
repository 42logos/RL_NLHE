"""PyQt6-based demo GUI for playing a single NLHE hand.

This widget-heavy rewrite organises the table in a more conventional poker
layout.  Community cards and pot sit in the middle while six ``PlayerPanel``
widgets surround the table.  Each panel shows stack, bet, status, hole cards
and last action with colour cues for folds, calls, raises and all-ins.  The
acting seat receives a yellow border and the action bar exposes a raise slider
that specifies the absolute amount to raise to.

The demo is intentionally lightweight but demonstrates how to drive the
``NLHEngine`` interactively from a GUI application.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtWidgets

from ..agents.tamed_random import TamedRandomAgent
from ..core.cards import rank_of, suit_of
from ..core.engine import NLHEngine
from ..core.types import Action, ActionType, GameState, PlayerState


# ----- card helpers -------------------------------------------------------
RSTR = {11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT = ["♣", "♦", "♥", "♠"]


def card_str(c: int) -> str:
    r = rank_of(c); s = suit_of(c)
    rs = str(r) if r <= 10 else RSTR[r]
    return f"{rs}{SUIT[s]}"


def cards_str(cards: List[int]) -> str:
    return " ".join(card_str(c) for c in cards)


# ----- player widget ------------------------------------------------------
class PlayerPanel(QtWidgets.QFrame):
    """Visual representation of a single player's public state."""

    def __init__(self, seat: int) -> None:
        super().__init__()
        self.seat = seat
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self.info = QtWidgets.QLabel(f"Seat {seat}")
        self.hole = QtWidgets.QLabel("?? ??")
        self.last = QtWidgets.QLabel("")
        lay.addWidget(self.info)
        lay.addWidget(self.hole)
        lay.addWidget(self.last)

        self._opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)

    def update(self, p: PlayerState, hero: bool,
               active: bool, last: Optional[Tuple[int, int]]) -> None:
        hole = cards_str(list(p.hole)) if hero and p.hole else "?? ??"
        self.hole.setText(hole)
        self.info.setText(
            f"Stack {p.stack} | Bet {p.bet} | Cont {p.cont} | {p.status}"
        )

        last_txt = ""
        bg = "white"
        if p.status == "folded":
            bg = "#dddddd"
            last_txt = "fold"
        elif p.status == "allin":
            bg = "#ffddaa"
            last_txt = "all-in"
        elif last is not None:
            aid, amt = last
            if aid == 1:
                last_txt = "check"
            elif aid == 2:
                last_txt = "call"; bg = "#cce0ff"
            elif aid == 3:
                last_txt = f"raise to {amt}"; bg = "#c4f5c4"
            else:
                last_txt = "fold"; bg = "#dddddd"
        self.last.setText(last_txt)

        border = "yellow" if active else "black"
        self.setStyleSheet(
            f"border: 2px solid {border}; color: black; background-color: {bg};"
        )
        self._opacity.setOpacity(1.0 if active else 0.6)


# ----- main window --------------------------------------------------------
class NLHEGui(QtWidgets.QMainWindow):
    """Play a single 6-max NLHE hand against basic random agents."""

    def __init__(self, hero_seat: int = 0, seed: int = 42) -> None:
        super().__init__()
        self.setWindowTitle("NLHE 6-Max GUI")

        self.hero_seat = hero_seat
        self.seed_val = seed
        self.rng = random.Random(seed)
        self.engine = NLHEngine(sb=1, bb=2, start_stack=100, rng=self.rng)
        self.agents: List[TamedRandomAgent | None] = [
            TamedRandomAgent(self.rng) for _ in range(self.engine.N)
        ]
        self.agents[hero_seat] = None  # human
        self.state: GameState = self.engine.reset_hand(button=0)

        self._create_widgets()
        self._update_view()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self._play_loop)
        self.timer.start()

    # ----- construction --------------------------------------------------
    def _create_widgets(self) -> None:
        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        # table layout
        grid = QtWidgets.QGridLayout(); main.addLayout(grid)
        pos = {0: (2, 1), 1: (1, 2), 2: (0, 2), 3: (0, 1), 4: (0, 0), 5: (1, 0)}

        self.player_panels: List[PlayerPanel] = []
        for seat in range(self.engine.N):
            panel = PlayerPanel(seat)
            self.player_panels.append(panel)
            grid.addWidget(panel, *pos[seat])

        center = QtWidgets.QWidget(); grid.addWidget(center, 1, 1)
        cl = QtWidgets.QVBoxLayout(center)
        cl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.board_label = QtWidgets.QLabel("Board: (preflop)")
        self.pot_label = QtWidgets.QLabel("Pot: 0")
        cl.addWidget(self.board_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self.pot_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # action bar
        btn_layout = QtWidgets.QHBoxLayout(); main.addLayout(btn_layout)
        self.action_buttons: Dict[str, QtWidgets.QPushButton] = {}
        for name in ["FOLD", "CHECK", "CALL", "RAISE"]:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _, n=name: self._on_action(n))
            btn_layout.addWidget(btn)
            self.action_buttons[name] = btn

        self.raise_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.raise_slider.setFixedWidth(120)
        self.raise_slider.valueChanged.connect(
            lambda v: self.raise_edit.setText(str(v))
        )
        btn_layout.addWidget(self.raise_slider)

        self.raise_edit = QtWidgets.QLineEdit()
        self.raise_edit.setFixedWidth(60)
        self.raise_edit.setPlaceholderText("raise to")
        btn_layout.addWidget(self.raise_edit)

        self.raise_info = QtWidgets.QLabel("")
        btn_layout.addWidget(self.raise_info)

        self.status_label = QtWidgets.QLabel("")
        main.addWidget(self.status_label)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(120)
        main.addWidget(self.log)

        seed_row = QtWidgets.QHBoxLayout(); main.addLayout(seed_row)
        seed_row.addWidget(QtWidgets.QLabel("Seed:"))
        self.seed_edit = QtWidgets.QLineEdit(str(self.seed_val))
        self.seed_edit.setFixedWidth(80)
        seed_row.addWidget(self.seed_edit)
        self.next_hand_btn = QtWidgets.QPushButton("Next Hand")
        self.next_hand_btn.setEnabled(False)
        self.next_hand_btn.clicked.connect(self._start_next_hand)
        seed_row.addWidget(self.next_hand_btn)

    # ----- helpers -------------------------------------------------------
    def _last_action(self, seat: int) -> Optional[Tuple[int, int]]:
        for sid, aid, amt, _ in reversed(self.state.actions_log):
            if sid == seat:
                return aid, amt
        return None

    def _log_action(self, seat: int, action: Action) -> None:
        if action.kind == ActionType.RAISE_TO:
            msg = f"Seat {seat} raises to {action.amount}"
        elif action.kind == ActionType.CALL:
            msg = f"Seat {seat} calls"
        elif action.kind == ActionType.CHECK:
            msg = f"Seat {seat} checks"
        else:
            msg = f"Seat {seat} folds"
        self.log.appendPlainText(msg)

    def _update_view(self) -> None:
        # board and pot
        if self.state.board:
            self.board_label.setText(f"Board: {cards_str(self.state.board)}")
        else:
            self.board_label.setText("Board: (preflop)")
        self.pot_label.setText(f"Pot: {self.state.pot}")

        # players
        for i, pnl in enumerate(self.player_panels):
            p = self.state.players[i]
            pnl.update(p, i == self.hero_seat,
                       self.state.next_to_act == i,
                       self._last_action(i))

        # status and action availability
        if self.state.next_to_act is not None:
            seat = self.state.next_to_act
            to_call = self.state.current_bet - self.state.players[seat].bet
            self.status_label.setText(
                f"Seat {seat} to act - to call {to_call}"
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
        self.raise_slider.setEnabled(raise_allowed)

        if raise_allowed:
            self.min_raise_to = getattr(info, "min_raise_to", None)
            self.max_raise_to = getattr(info, "max_raise_to", None)
            min_amt = self.min_raise_to if self.min_raise_to is not None else 0
            max_amt = self.max_raise_to if self.max_raise_to is not None else min_amt
            self.raise_slider.setMinimum(min_amt)
            self.raise_slider.setMaximum(max_amt)
            self.raise_slider.setValue(min_amt)
            self.raise_edit.setText(str(min_amt))
            self.raise_info.setText(f"[{min_amt}-{max_amt}]")
        else:
            self.raise_info.setText("")

    # ----- gameplay loop -------------------------------------------------
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
            return  # wait for user

        agent = self.agents[seat]
        assert agent is not None
        action = agent.act(self.engine, self.state, seat)
        self._log_action(seat, action)
        self.state, done, rewards, _ = self.engine.step(self.state, action)
        if done:
            self._end_hand(rewards); return
        if self.state.next_to_act is None:
            done, rewards = self.engine.advance_round_if_needed(self.state)
            if done:
                self._end_hand(rewards); return
        self._update_view()

    # ----- user actions --------------------------------------------------
    def _on_action(self, name: str) -> None:
        if self.state.next_to_act != self.hero_seat:
            return
        if name == "FOLD":
            action = Action(ActionType.FOLD)
        elif name == "CHECK":
            action = Action(ActionType.CHECK)
        elif name == "CALL":
            action = Action(ActionType.CALL)
        elif name == "RAISE":
            try:
                amt = int(self.raise_edit.text())
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "Invalid", "Enter raise amount")
                return
            if self.min_raise_to is not None and amt < self.min_raise_to:
                QtWidgets.QMessageBox.critical(
                    self, "Invalid", f"Minimum raise is {self.min_raise_to}"
                )
                return
            if self.max_raise_to is not None and amt > self.max_raise_to:
                QtWidgets.QMessageBox.critical(
                    self, "Invalid", f"Maximum raise is {self.max_raise_to}"
                )
                return
            action = Action(ActionType.RAISE_TO, amount=amt)
        else:
            return

        self._log_action(self.hero_seat, action)
        self.state, done, rewards, _ = self.engine.step(self.state, action)
        if done:
            self._end_hand(rewards); return
        if self.state.next_to_act is None:
            done, rewards = self.engine.advance_round_if_needed(self.state)
            if done:
                self._end_hand(rewards); return
        self._update_view()

    # ----- end -----------------------------------------------------------
    def _end_hand(self, rewards: List[int]) -> None:
        for i, pnl in enumerate(self.player_panels):
            p = self.state.players[i]
            hole = cards_str(list(p.hole)) if p.hole else "?? ??"
            pnl.hole.setText(hole)
        msg = "\n".join(f"Seat {i}: {r}" for i, r in enumerate(rewards))
        QtWidgets.QMessageBox.information(self, "Hand complete", msg)
        for btn in self.action_buttons.values():
            btn.setEnabled(False)
        self.raise_edit.setEnabled(False)
        self.raise_slider.setEnabled(False)
        self.status_label.setText("Hand complete")
        self.timer.stop()
        self.next_hand_btn.setEnabled(True)

    def _start_next_hand(self) -> None:
        try:
            seed = int(self.seed_edit.text())
        except ValueError:
            seed = random.randrange(1 << 30)
            self.seed_edit.setText(str(seed))
        self.seed_val = seed
        self.rng.seed(seed)
        button = (self.state.button + 1) % self.engine.N
        self.state = self.engine.reset_hand(button=button)
        self.log.clear()
        for btn in self.action_buttons.values():
            btn.setEnabled(True)
        self.raise_edit.setEnabled(True)
        self.raise_slider.setEnabled(True)
        self.next_hand_btn.setEnabled(False)
        self.timer.start()
        self._update_view()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NLHE GUI demo")
    parser.add_argument("--hero-seat", type=int, default=0,
                        help="human seat index (0-5)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    gui = NLHEGui(hero_seat=args.hero_seat, seed=args.seed)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()

