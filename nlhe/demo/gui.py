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

from typing import Dict, List, Optional, Tuple
from pathlib import Path

from PyQt6 import QtCore, QtWidgets, QtGui

from ..core.types import Action, ActionType, GameState
from .controller import GameController
from .widgets import BoardWidget, PlayerPanel


class NLHEGui(QtWidgets.QMainWindow):
    """Play a single 6-max NLHE hand against basic random agents."""

    def __init__(self, hero_seat: int = 0, seed: int = 42) -> None:
        super().__init__()
        self.setWindowTitle("NLHE 6-Max GUI")
        self.hero_seat = hero_seat

        # load application wide stylesheet
        style_file = Path(__file__).with_name("styles.qss")
        if style_file.exists():
            self.setStyleSheet(style_file.read_text())

        self.controller = GameController(hero_seat=hero_seat, seed=seed)
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.hand_finished.connect(self._end_hand)
        self.controller.action_logged.connect(self._log_action)

        self._create_widgets()
        self._on_state_changed(self.controller.state)

    # ----- construction --------------------------------------------------
    def _create_widgets(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        grid = QtWidgets.QGridLayout()
        main.addLayout(grid)
        pos = {0: (2, 1), 1: (1, 2), 2: (0, 2), 3: (0, 1), 4: (0, 0), 5: (1, 0)}

        self.player_panels: List[PlayerPanel] = []
        for seat in range(self.controller.engine.N):
            panel = PlayerPanel(seat)
            self.player_panels.append(panel)
            grid.addWidget(panel, *pos[seat])

        center = QtWidgets.QWidget()
        grid.addWidget(center, 1, 1)
        cl = QtWidgets.QVBoxLayout(center)
        cl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.board = BoardWidget()
        cl.addWidget(self.board)
        self.pot_label = QtWidgets.QLabel("Pot: 0")
        cl.addWidget(self.pot_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        btn_layout = QtWidgets.QHBoxLayout()
        main.addLayout(btn_layout)
        self.action_buttons: Dict[str, QtWidgets.QPushButton] = {}

        icons_dir = Path(__file__).with_name("assets") / "icons"
        btn_specs = {
            "FOLD": ("fold.svg", "Forfeit the hand"),
            "CHECK": ("check.svg", "Pass action without betting"),
            "CALL": ("call.svg", "Match the current bet"),
            "RAISE": ("raise.svg", "Increase the bet amount"),
        }

        for name, (icon_file, tip) in btn_specs.items():
            btn = QtWidgets.QPushButton(name)
            btn.setObjectName(f"{name.lower()}-button")
            btn.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_StyledBackground, True
            )
            btn.setIcon(QtGui.QIcon(str(icons_dir / icon_file)))
            btn.setIconSize(QtCore.QSize(16, 16))
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _, n=name: self._on_action(n))

            # simple press animation to give visual feedback
            effect = QtWidgets.QGraphicsOpacityEffect(btn)
            btn.setGraphicsEffect(effect)
            anim = QtCore.QSequentialAnimationGroup(btn)
            fade_out = QtCore.QPropertyAnimation(effect, b"opacity")
            fade_out.setDuration(100)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.3)
            fade_in = QtCore.QPropertyAnimation(effect, b"opacity")
            fade_in.setDuration(150)
            fade_in.setStartValue(0.3)
            fade_in.setEndValue(1.0)
            anim.addAnimation(fade_out)
            anim.addAnimation(fade_in)
            btn.pressed.connect(anim.start)
            btn._press_anim = anim  # keep reference

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
        self.status_label.setObjectName("status-label")
        self.status_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_StyledBackground, True
        )
        main.addWidget(self.status_label)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(120)
        main.addWidget(self.log)

        seed_row = QtWidgets.QHBoxLayout()
        main.addLayout(seed_row)
        seed_row.addWidget(QtWidgets.QLabel("Seed:"))
        self.seed_edit = QtWidgets.QLineEdit(str(self.controller.seed_val))
        self.seed_edit.setFixedWidth(80)
        seed_row.addWidget(self.seed_edit)
        self.next_hand_btn = QtWidgets.QPushButton("Next Hand")
        self.next_hand_btn.setEnabled(False)
        self.next_hand_btn.clicked.connect(self._start_next_hand)
        seed_row.addWidget(self.next_hand_btn)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # pragma: no cover - GUI
        """Refresh layout when the window size changes."""
        super().resizeEvent(event)
        self._update_view()

    # ----- helpers -------------------------------------------------------
    def _last_action(self, seat: int) -> Optional[Tuple[int, int]]:
        for sid, aid, amt, _ in reversed(self.controller.state.actions_log):
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

    def _on_state_changed(self, _state: GameState) -> None:
        self._update_view()

    def _update_view(self) -> None:
        state = self.controller.state
        self.board.update(state.board)
        self.pot_label.setText(f"Pot: {state.pot}")

        for i, pnl in enumerate(self.player_panels):
            p = state.players[i]
            pnl.update(
                p,
                i == self.hero_seat,
                state.next_to_act == i,
                self._last_action(i),
            )

        if state.next_to_act is not None:
            seat = state.next_to_act
            to_call = state.current_bet - state.players[seat].bet
            self.status_label.setText(
                f"Seat {seat} to act - to call {to_call}"
            )
        else:
            self.status_label.setText("Waiting for round advance...")

        info = self.controller.engine.legal_actions(state)
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

    # ----- user actions --------------------------------------------------
    def _on_action(self, name: str) -> None:
        if self.controller.state.next_to_act != self.hero_seat:
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
                QtWidgets.QMessageBox.critical(
                    self, "Invalid", "Enter raise amount"
                )
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

        self.controller.submit_action(action)

    # ----- end -----------------------------------------------------------
    def _end_hand(self, rewards: List[int]) -> None:
        state = self.controller.state
        for i, pnl in enumerate(self.player_panels):
            p = state.players[i]
            for j, lbl in enumerate(pnl.card_labels):
                lbl.set_card(p.hole[j] if p.hole and j < len(p.hole) else None)
        msg = "\n".join(f"Seat {i}: {r}" for i, r in enumerate(rewards))
        QtWidgets.QMessageBox.information(self, "Hand complete", msg)
        for btn in self.action_buttons.values():
            btn.setEnabled(False)
        self.raise_edit.setEnabled(False)
        self.raise_slider.setEnabled(False)
        self.status_label.setText("Hand complete")
        self.seed_edit.setText(str(self.controller.seed_val))
        self.next_hand_btn.setEnabled(True)

    def _start_next_hand(self) -> None:
        try:
            seed = int(self.seed_edit.text())
        except ValueError:
            seed = None
        self.controller.start_next_hand(seed)
        self.seed_edit.setText(str(self.controller.seed_val))
        self.log.clear()
        for btn in self.action_buttons.values():
            btn.setEnabled(True)
        self.raise_edit.setEnabled(True)
        self.raise_slider.setEnabled(True)
        self.next_hand_btn.setEnabled(False)
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
