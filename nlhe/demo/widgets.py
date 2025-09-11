from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path

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


class BetChip(QtWidgets.QWidget):
    """Chip-style widget showing the player's current bet with animation."""

    _pix_cache: Dict[str, QtGui.QPixmap] = {}
    COLOR_LEVELS: List[Tuple[int, str]] = [
        # thresholds ordered from highest to lowest
        (500, "purple"),
        (100, "black"),
        (25, "green"),
        (5, "blue"),
        (0, "red"),
    ]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._amt = 0
        self._color = "red"
        self._base = 32
        self._max = int(self._base * 1.5)
        self._apply_size(self._base)
        self._opacity = 1.0
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(8)
        shadow.setOffset(0, 2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 160))
        self.setGraphicsEffect(shadow)

    def _apply_size(self, s: int) -> None:
        self._size = s
        self.setFixedSize(s, s)
        self.update()

    @property
    def max_size(self) -> int:
        return self._max

    def set_amount(self, amt: int) -> None:
        self._amt = amt
        self.setVisible(amt > 0)
        self._color = self.color_for_bet(amt)
        self.update()

    def _get_opacity(self) -> float:
        return self._opacity

    def _set_opacity(self, value: float) -> None:
        self._opacity = value
        self.update()

    opacity = QtCore.pyqtProperty(float, fget=_get_opacity, fset=_set_opacity)

    @classmethod
    def color_for_bet(cls, amt: int) -> str:
        """Return chip color name for a given bet amount."""
        for threshold, color in cls.COLOR_LEVELS:
            if amt >= threshold:
                return color
        return "red"

    @classmethod
    def _chip_pixmap(cls, color: str) -> QtGui.QPixmap:
        if color not in cls._pix_cache:
            path = Path(__file__).with_name("assets").joinpath("chips", f"chip_{color}.png")
            pix = QtGui.QPixmap(str(path))
            if pix.isNull():
                size = 64
                pix = QtGui.QPixmap(size, size)
                pix.fill(QtCore.Qt.GlobalColor.transparent)
                p = QtGui.QPainter(pix)
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                p.setBrush(QtGui.QColor(color))
                p.drawEllipse(0, 0, size - 1, size - 1)
                p.end()
            cls._pix_cache[color] = pix
        return cls._pix_cache[color]

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:  # pragma: no cover - GUI paint
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setOpacity(self._opacity)
        pix = self._chip_pixmap(self._color).scaled(
            self._size,
            self._size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        p.drawPixmap(self.rect(), pix)
        font = p.font()
        font.setBold(True)
        font.setPointSize(max(8, int(self._size / 2)))
        p.setFont(font)
        p.setPen(QtGui.QColor("white"))
        p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, str(self._amt))
        p.end()

    def animate(self) -> None:
        self._opacity = 0.0
        group = QtCore.QParallelAnimationGroup(self)

        fade = QtCore.QPropertyAnimation(self, b"opacity")
        fade.setDuration(500)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)

        scale = QtCore.QVariantAnimation(self)
        scale.setDuration(500)
        scale.setStartValue(self._max)
        scale.setEndValue(self._base)
        scale.valueChanged.connect(lambda v: self._apply_size(int(v)))

        group.addAnimation(fade)
        group.addAnimation(scale)
        group.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)


class PlayerPanel(QtWidgets.QFrame):
    """Visual representation of a single player's public state."""

    def __init__(self, seat: int) -> None:
        super().__init__()
        self.seat = seat
        self.setObjectName("player-panel")
        # allow QSS background colors to fully apply
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        info_row = QtWidgets.QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(4)

        # Seat indicator shown once with text
        self.seat_label = QtWidgets.QLabel(f"Seat {seat}")
        self.seat_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.seat_label.setFixedHeight(24)
        self.seat_label.setObjectName("badge")
        self.seat_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_StyledBackground, True
        )
        info_row.addWidget(self.seat_label)

        info_row.addStretch(1)

        self.stack_label = QtWidgets.QLabel("Stack 0")
        self.stack_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        info_row.addWidget(self.stack_label)

        self.bet_chip = BetChip()
        # Reserve space for bet chip including animation oversize
        self.bet_box = QtWidgets.QWidget()
        self.bet_box.setFixedSize(self.bet_chip.max_size, self.bet_chip.max_size)
        bet_lay = QtWidgets.QHBoxLayout(self.bet_box)
        bet_lay.setContentsMargins(0, 0, 0, 0)
        bet_lay.addStretch(1)
        bet_lay.addWidget(self.bet_chip)
        info_row.addWidget(self.bet_box)

        lay.addLayout(info_row)

        cards_row = QtWidgets.QHBoxLayout()
        self.card_labels: List[CardWidget] = [CardWidget(), CardWidget()]
        for lbl in self.card_labels:
            cards_row.addWidget(lbl)
        lay.addLayout(cards_row)

        self.last = QtWidgets.QLabel("")
        self.last.setObjectName("status-label")
        self.last.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_StyledBackground, True
        )
        self.last.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.last.setFixedHeight(24)
        lay.addWidget(self.last)

        margins = lay.contentsMargins()
        min_width = (
            self.seat_label.sizeHint().width()
            + self.stack_label.sizeHint().width()
            + self.bet_box.width()
            + info_row.spacing() * 2
            + margins.left()
            + margins.right()
        )
        self.setMinimumWidth(min_width)

        self._last_bet = 0

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
        self.bet_chip.set_amount(p.bet)
        if p.bet > self._last_bet:
            self.bet_chip.animate()
        self._last_bet = p.bet

        last_txt = ""
        state = "default"
        if p.status == "folded":
            state = "folded"
            last_txt = "fold"
        elif p.status == "allin":
            state = "allin"
            last_txt = "all-in"
        elif last is not None:
            aid, amt = last
            if aid == 1:
                last_txt = "check"
            elif aid == 2:
                state = "called"
                last_txt = "call"
            elif aid == 3:
                state = "raised"
                last_txt = f"raise to {amt}"
            else:
                state = "folded"
                last_txt = "fold"
        self.last.setText(last_txt)
        self.last.setProperty("state", state)
        self.last.style().unpolish(self.last)
        self.last.style().polish(self.last)

        self.setProperty("active", active)
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        # call QWidget.update to refresh after re-polishing style
        super().update()


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
