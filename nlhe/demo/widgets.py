from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from ..core.cards import rank_of, suit_of

RSTR = {11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT = ["♣", "♦", "♥", "♠"]


def card_to_pixmap(card: int, width: int = 40, height: int = 60) -> QtGui.QPixmap:
    """Return a simple pixmap for ``card`` with rank/suit text."""
    pix = QtGui.QPixmap(width, height)
    pix.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(QtGui.QColor("black")))
    painter.drawRect(0, 0, width - 1, height - 1)
    r = rank_of(card)
    s = suit_of(card)
    rank_txt = str(r) if r <= 10 else RSTR[r]
    suit_txt = SUIT[s]
    color = QtGui.QColor("red") if s in (1, 2) else QtGui.QColor("black")
    painter.setPen(color)
    font = QtGui.QFont("Arial", 14)
    painter.setFont(font)
    painter.drawText(5, 20, rank_txt)
    painter.drawText(width // 2 - 7, height // 2 + 10, suit_txt)
    painter.end()
    return pix


def card_back_pixmap(width: int = 40, height: int = 60) -> QtGui.QPixmap:
    pix = QtGui.QPixmap(width, height)
    pix.fill(QtGui.QColor("#1e5799"))
    painter = QtGui.QPainter(pix)
    painter.setPen(QtGui.QPen(QtGui.QColor("white")))
    painter.drawRect(0, 0, width - 1, height - 1)
    painter.end()
    return pix


def chip_pixmap(amount: int, diameter: int = 24) -> QtGui.QPixmap:
    pix = QtGui.QPixmap(diameter, diameter)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setBrush(QtGui.QColor("#d18b47"))
    painter.setPen(QtGui.QPen(QtGui.QColor("black")))
    painter.drawEllipse(0, 0, diameter - 1, diameter - 1)
    painter.setPen(QtGui.QColor("white"))
    font = QtGui.QFont("Arial", 10)
    painter.setFont(font)
    painter.drawText(pix.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, str(amount))
    painter.end()
    return pix


class CardWidget(QtWidgets.QLabel):
    """Label showing a single card image."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(40, 60)
        self.setPixmap(card_back_pixmap())

    def set_card(self, card: Optional[int]) -> None:
        if card is None:
            self.setPixmap(card_back_pixmap())
        else:
            self.setPixmap(card_to_pixmap(card))


class BoardWidget(QtWidgets.QWidget):
    """Display community cards."""

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.cards: List[CardWidget] = [CardWidget() for _ in range(5)]
        for w in self.cards:
            lay.addWidget(w)

    def set_cards(self, cards: Sequence[int]) -> None:
        for i, w in enumerate(self.cards):
            w.set_card(cards[i] if i < len(cards) else None)


class PlayerPanel(QtWidgets.QFrame):
    """Visual representation of a single player's public state."""

    def __init__(self, seat: int) -> None:
        super().__init__()
        self.seat = seat
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self.seat_label = QtWidgets.QLabel(f"Seat {seat}")
        lay.addWidget(self.seat_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        card_row = QtWidgets.QHBoxLayout()
        self.hole: List[CardWidget] = [CardWidget(), CardWidget()]
        for c in self.hole:
            card_row.addWidget(c)
        lay.addLayout(card_row)

        self.info = QtWidgets.QLabel("")
        self.last = QtWidgets.QLabel("")
        lay.addWidget(self.info)
        lay.addWidget(self.last)

        self.chip_label = QtWidgets.QLabel("")
        self.chip_label.setVisible(False)
        self._chip_opacity = QtWidgets.QGraphicsOpacityEffect(self.chip_label)
        self.chip_label.setGraphicsEffect(self._chip_opacity)
        lay.addWidget(self.chip_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._chip_anim = QtCore.QPropertyAnimation(self._chip_opacity, b"opacity")
        self._chip_anim.setDuration(500)

        self._opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)

    def _animate_chip(self) -> None:
        self._chip_anim.stop()
        self._chip_anim.setStartValue(0.0)
        self._chip_anim.setEndValue(1.0)
        self._chip_anim.start()

    def update(self, p: 'PlayerState', hero: bool,
               active: bool, last: Optional[Tuple[int, int]]) -> None:
        # hole cards
        if hero and p.hole:
            cards = list(p.hole)
            for i in range(2):
                self.hole[i].set_card(cards[i])
        else:
            for c in self.hole:
                c.set_card(None)

        self.info.setText(
            f"Stack {p.stack} | Bet {p.bet} | Cont {p.cont} | {p.status}"
        )

        # last action and chip graphic
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

        self.chip_label.setVisible(p.bet > 0)
        if p.bet > 0:
            self.chip_label.setPixmap(chip_pixmap(p.bet))
            if last is not None and last[0] in (2, 3):
                self._animate_chip()

        border = "#280401" if active else "black"
        self.setStyleSheet(
            f"border: 2px solid {border}; color: black; background-color: {bg};"
        )
        self._opacity.setOpacity(1.0 if active else 0.6)

    def reveal(self, cards: Sequence[int]) -> None:
        for i in range(2):
            self.hole[i].set_card(cards[i] if i < len(cards) else None)
