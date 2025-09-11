from PyQt6 import QtGui, QtCore
from pathlib import Path


def make_base(size: int = 64) -> QtGui.QPixmap:
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setBrush(QtGui.QColor("white"))
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.drawEllipse(0, 0, size - 1, size - 1)
    p.end()
    return pix


def make_highlight(size: int = 64) -> QtGui.QPixmap:
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen_outer = QtGui.QPen(QtGui.QColor("white"), 2)
    p.setPen(pen_outer)
    p.drawEllipse(3, 3, size - 7, size - 7)
    pen_arc = QtGui.QPen(QtGui.QColor("white"), 3)
    p.setPen(pen_arc)
    for ang in range(0, 360, 45):
        p.drawArc(3, 3, size - 7, size - 7, ang * 16, 10 * 16)
    p.end()
    return pix


def main() -> None:
    base = make_base()
    hl = make_highlight()
    out_dir = Path(__file__).parent
    base.save(str(out_dir / "chip_base.png"))
    hl.save(str(out_dir / "chip_highlight.png"))


if __name__ == "__main__":
    app = QtGui.QGuiApplication([])
    main()
