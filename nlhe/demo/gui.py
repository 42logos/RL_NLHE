
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

from PyQt6 import QtCore, QtGui, QtWidgets

from ..core.cards import rank_of, suit_of
from ..core.types import Action, ActionType, GameState, PlayerState
from .controller import GameController


# ----- card helpers -------------------------------------------------------
RSTR = {11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT = ["♣", "♦", "♥", "♠"]

CARD_SIZE = QtCore.QSize(50, 70)


def _card_text(c: int) -> Tuple[str, str, QtGui.QColor]:
    r = rank_of(c)
    s = suit_of(c)
    rs = str(r) if r <= 10 else RSTR[r]
    suit = SUIT[s]
    color = QtGui.QColor("red") if s in (1, 2) else QtGui.QColor("black")
    return rs, suit, color


def card_pixmap(c: int) -> QtGui.QPixmap:
    rs, suit, color = _card_text(c)
    pix = QtGui.QPixmap(CARD_SIZE)
    pix.fill(QtGui.QColor("white"))
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor("black")))
    p.drawRect(0, 0, CARD_SIZE.width() - 1, CARD_SIZE.height() - 1)
    font = p.font()
    font.setBold(True)
    p.setFont(font)
    p.setPen(color)
    p.drawText(pix.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, f"{rs}\n{suit}")
    p.end()
    return pix


def card_back_pixmap() -> QtGui.QPixmap:
    pix = QtGui.QPixmap(CARD_SIZE)
    pix.fill(QtGui.QColor("#0b5fa5"))
    p = QtGui.QPainter(pix)
    p.setPen(QtGui.QPen(QtGui.QColor("white")))
    p.drawRect(0, 0, CARD_SIZE.width() - 1, CARD_SIZE.height() - 1)
    p.end()
    return pix


def chip_pixmap() -> QtGui.QPixmap:
    size = 20
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setBrush(QtGui.QColor("#d43333"))
    p.drawEllipse(0, 0, size - 1, size - 1)
    p.setPen(QtGui.QPen(QtGui.QColor("white"), 2))
    p.drawEllipse(3, 3, size - 7, size - 7)
    p.end()
    return pix


# ----- player widget ------------------------------------------------------
class PlayerPanel(QtWidgets.QFrame):
    """Visual representation of a single player's public state."""

    def __init__(self, seat: int) -> None:
        super().__init__()
        self.seat = seat
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        info_row = QtWidgets.QHBoxLayout()
        self.seat_label = QtWidgets.QLabel(str(seat))
        self.seat_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.seat_label.setFixedSize(24, 24)
        self.seat_label.setStyleSheet(
            "border-radius:12px; background:#333333; color:white;"
        )
        info_row.addWidget(self.seat_label)
        self.stack_label = QtWidgets.QLabel("Stack 0")
        info_row.addWidget(self.stack_label)
        chip = QtWidgets.QLabel()
        chip.setPixmap(chip_pixmap())
        info_row.addWidget(chip)
        self.bet_label = QtWidgets.QLabel("Bet 0")
        info_row.addWidget(self.bet_label)
        lay.addLayout(info_row)

        cards_row = QtWidgets.QHBoxLayout()
        self.card_labels: List[QtWidgets.QLabel] = []
        for _ in range(2):
            lbl = QtWidgets.QLabel()
            lbl.setFixedSize(CARD_SIZE)
            lbl.setPixmap(card_back_pixmap())
            cards_row.addWidget(lbl)
            self.card_labels.append(lbl)
        lay.addLayout(cards_row)

        self.last = QtWidgets.QLabel("")
        lay.addWidget(self.last)

        self._opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._bet_effect = QtWidgets.QGraphicsOpacityEffect(self.bet_label)
        self.bet_label.setGraphicsEffect(self._bet_effect)
        self._last_bet = 0

    def _animate_bet(self) -> None:
        anim = QtCore.QPropertyAnimation(self._bet_effect, b"opacity")
        anim.setDuration(600)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def update(self, p: PlayerState, hero: bool,
               active: bool, last: Optional[Tuple[int, int]]) -> None:
        if hero and p.hole:
            holes = list(p.hole)
        else:
            holes = []
        for i, lbl in enumerate(self.card_labels):
            if i < len(holes):
                lbl.setPixmap(card_pixmap(holes[i]))
            else:
                lbl.setPixmap(card_back_pixmap())

        self.stack_label.setText(f"Stack {p.stack}")
        self.bet_label.setText(f"Bet {p.bet}")
        if p.bet > self._last_bet:
            self._animate_bet()
        self._last_bet = p.bet

        last_txt = ""
        bg = "#fcdcda"
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

        border = "#280401" if active else "black"
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

        self.controller = GameController(hero_seat=hero_seat, seed=seed)
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.hand_finished.connect(self._end_hand)
        self.controller.action_logged.connect(self._log_action)

        self._create_widgets()
        self._on_state_changed(self.controller.state)

    # ----- construction --------------------------------------------------
    def _create_widgets(self) -> None:
        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        # table layout
        grid = QtWidgets.QGridLayout(); main.addLayout(grid)
        pos = {0: (2, 1), 1: (1, 2), 2: (0, 2), 3: (0, 1), 4: (0, 0), 5: (1, 0)}

        self.player_panels: List[PlayerPanel] = []
        for seat in range(self.controller.engine.N):
            panel = PlayerPanel(seat)
            self.player_panels.append(panel)
            grid.addWidget(panel, *pos[seat])

        center = QtWidgets.QWidget(); grid.addWidget(center, 1, 1)
        cl = QtWidgets.QVBoxLayout(center)
        cl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        board_row = QtWidgets.QHBoxLayout()
        self.board_cards: List[QtWidgets.QLabel] = []
        for _ in range(5):
            lbl = QtWidgets.QLabel()
            lbl.setFixedSize(CARD_SIZE)
            lbl.setPixmap(card_back_pixmap())
            board_row.addWidget(lbl)
            self.board_cards.append(lbl)
        cl.addLayout(board_row)
        self.pot_label = QtWidgets.QLabel("Pot: 0")
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
        self.seed_edit = QtWidgets.QLineEdit(str(self.controller.seed_val))
        self.seed_edit.setFixedWidth(80)
        seed_row.addWidget(self.seed_edit)
        self.next_hand_btn = QtWidgets.QPushButton("Next Hand")
        self.next_hand_btn.setEnabled(False)
        self.next_hand_btn.clicked.connect(self._start_next_hand)
        seed_row.addWidget(self.next_hand_btn)

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
        # board and pot
        state = self.controller.state
        for i, lbl in enumerate(self.board_cards):
            if i < len(state.board):
                lbl.setPixmap(card_pixmap(state.board[i]))
            else:
                lbl.setPixmap(card_back_pixmap())
        self.pot_label.setText(f"Pot: {state.pot}")

        # players
        for i, pnl in enumerate(self.player_panels):
            p = state.players[i]
            pnl.update(p, i == self.hero_seat,
                       state.next_to_act == i,
                       self._last_action(i))

        # status and action availability
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

        self.controller.submit_action(action)

    # ----- end -----------------------------------------------------------
    def _end_hand(self, rewards: List[int]) -> None:
        state = self.controller.state
        for i, pnl in enumerate(self.player_panels):
            p = state.players[i]
            for j, lbl in enumerate(pnl.card_labels):
                if p.hole and j < len(p.hole):
                    lbl.setPixmap(card_pixmap(p.hole[j]))
                else:
                    lbl.setPixmap(card_back_pixmap())
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

