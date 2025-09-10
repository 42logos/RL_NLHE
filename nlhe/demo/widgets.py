from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from ..core.cards import rank_of, suit_of
from ..core.types import PlayerState


class CardWidget(QtWidgets.QLabel):
    """QLabel capable of rendering a playing card face or back."""

    RSTR = {11: "J", 12: "Q", 13: "K", 14: "A"}
    SUIT = ["♣", "♦", "♥", "♠"]
    SIZE = QtCore.QSize(50, 70)

    _cache: Dict[int, QtGui.QPixmap] = {}
    _back: Optional[QtGui.QPixmap] = None

    @classmethod
    def _text(cls, c: int) -> Tuple[str, str, QtGui.QColor]:
        r = rank_of(c)
        s = suit_of(c)
        rs = str(r) if r <= 10 else cls.RSTR[r]
        suit = cls.SUIT[s]
        color = QtGui.QColor("red") if s in (1, 2) else QtGui.QColor("black")
        return rs, suit, color

    @classmethod
    def face_pixmap(cls, c: int) -> QtGui.QPixmap:
        if c not in cls._cache:
            rs, suit, color = cls._text(c)
            pix = QtGui.QPixmap(cls.SIZE)
            pix.fill(QtGui.QColor("white"))
            p = QtGui.QPainter(pix)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setPen(QtGui.QPen(QtGui.QColor("black")))
            p.drawRect(0, 0, cls.SIZE.width() - 1, cls.SIZE.height() - 1)
            font = p.font()
            font.setBold(True)
            p.setFont(font)
            p.setPen(color)
            p.drawText(pix.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, f"{rs}\n{suit}")
            p.end()
            cls._cache[c] = pix
        return cls._cache[c]

    @classmethod
    def back_pixmap(cls) -> QtGui.QPixmap:
        if cls._back is None:
            pix = QtGui.QPixmap(cls.SIZE)
            pix.fill(QtGui.QColor("#0b5fa5"))
            p = QtGui.QPainter(pix)
            p.setPen(QtGui.QPen(QtGui.QColor("white")))
            p.drawRect(0, 0, cls.SIZE.width() - 1, cls.SIZE.height() - 1)
            p.end()
            cls._back = pix
        return cls._back

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.SIZE)
        self.set_card(None)

    def set_card(self, card: Optional[int]) -> None:
        self.setPixmap(self.back_pixmap() if card is None else self.face_pixmap(card))


class ChipLabel(QtWidgets.QLabel):
    """Small chip icon used in player panels."""

    _pix: Optional[QtGui.QPixmap] = None

    @classmethod
    def pixmap(cls) -> QtGui.QPixmap:
        if cls._pix is None:
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
            cls._pix = pix
        return cls._pix

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setPixmap(self.pixmap())


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
            "border-radius:12px; background:#333333; color:white;",
        )
        info_row.addWidget(self.seat_label)
        self.seat_text = QtWidgets.QLabel(f"Seat {seat}")
        info_row.addWidget(self.seat_text)
        self.stack_label = QtWidgets.QLabel("Stack 0")
        info_row.addWidget(self.stack_label)
        info_row.addWidget(ChipLabel())
        self.round_label = QtWidgets.QLabel("Round 0")
        info_row.addWidget(self.round_label)
        lay.addLayout(info_row)

        cards_row = QtWidgets.QHBoxLayout()
        self.card_labels: List[CardWidget] = [CardWidget(), CardWidget()]
        for lbl in self.card_labels:
            cards_row.addWidget(lbl)
        lay.addLayout(cards_row)

        self.last = QtWidgets.QLabel("")
        lay.addWidget(self.last)

        self._opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._bet_effect = QtWidgets.QGraphicsOpacityEffect(self.round_label)
        self.round_label.setGraphicsEffect(self._bet_effect)
        self._last_bet = 0

    def _animate_bet(self) -> None:
        anim = QtCore.QPropertyAnimation(self._bet_effect, b"opacity")
        anim.setDuration(600)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def update(
        self,
        p: PlayerState,
        hero: bool,
        active: bool,
        last: Optional[Tuple[int, int]],
    ) -> None:
        holes = list(p.hole) if hero and p.hole else []
        for i, lbl in enumerate(self.card_labels):
            lbl.set_card(holes[i] if i < len(holes) else None)

        self.stack_label.setText(f"Stack {p.stack}")
        self.round_label.setText(f"Round {p.bet}")
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
            f"border: 2px solid {border}; color: black; background-color: {bg};",
        )
        self._opacity.setOpacity(1.0 if active else 0.6)


class BoardWidget(QtWidgets.QWidget):
    """Display for community cards and pot."""

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.cards: List[CardWidget] = [CardWidget() for _ in range(5)]
        for c in self.cards:
            lay.addWidget(c)

    def update(self, board: List[int]) -> None:
        for i, c in enumerate(board):
            self.cards[i].set_card(c)
        for j in range(len(board), 5):
            self.cards[j].set_card(None)
