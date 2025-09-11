import math
from pathlib import Path
from PIL import Image, ImageDraw

# Directory where images will be written (same directory as this file)
OUTPUT_DIR = Path(__file__).resolve().parent

# Base colors for chips
COLORS = {
    "red": "#d43333",
    "blue": "#3366cc",
    "green": "#3fa34d",
    "black": "#2c2c2c",
    "purple": "#8e44ad",
}

def draw_chip(color: str, size: int = 64) -> Image.Image:
    """Create a chip image of `size` pixels with the given base `color`.

    The chip is composed of a base circle with white highlights and small
    wedge-shaped accents similar to a traditional poker chip.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Base circle
    draw.ellipse((0, 0, size - 1, size - 1), fill=color)

    # Inner ring highlight
    ring_margin = int(size * 0.1)
    inner_bbox = (
        ring_margin,
        ring_margin,
        size - ring_margin - 1,
        size - ring_margin - 1,
    )
    ring_width = int(size * 0.08)
    draw.ellipse(inner_bbox, outline="white", width=ring_width)

    # Small arc highlights around the ring
    for ang in range(0, 360, 45):
        draw.arc(inner_bbox, start=ang, end=ang + 10, fill="white", width=ring_width)

    return img


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, color in COLORS.items():
        img = draw_chip(color)
        out_file = OUTPUT_DIR / f"chip_{name}.png"
        img.save(out_file)
        print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
