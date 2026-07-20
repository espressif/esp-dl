#!/usr/bin/env python3
"""Render PP-OCRv6 serial-log OCR results (PaddleOCR draw_ocr_box_txt layout).

Parses app_main log lines::

    text="...", score=0.xxxx, box=[x0,y0 x1,y1 x2,y2 x3,y3], det_score=0.xxxx

Usage::

    python tools/visualize_result.py \\
        --image main/pp_ocr_v6.jpg --log serial.log --output pp_ocr_v6_res.jpg

Dependencies: pillow, numpy, opencv-python; optional fontTools.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont


# Matches app_main log_ocr_result lines (any ESP_LOG prefix allowed).
_BOX_RE = re.compile(
    r'text="(?P<text>.*?)",\s*score=(?P<score>-?\d+(?:\.\d+)?),\s*'
    r"box=\[(?P<x0>-?\d+),(?P<y0>-?\d+)\s+"
    r"(?P<x1>-?\d+),(?P<y1>-?\d+)\s+"
    r"(?P<x2>-?\d+),(?P<y2>-?\d+)\s+"
    r"(?P<x3>-?\d+),(?P<y3>-?\d+)\],\s*"
    r"det_score=(?P<det_score>-?\d+(?:\.\d+)?)"
)

# Prefer PaddleOCR simfang.ttf for official look; then CJK system fonts.
# The selected font must cover every character in the OCR output.
_FONT_CANDIDATES = [
    os.path.expanduser("~/Downloads/PP-OCRv6-main/doc/fonts/simfang.ttf"),
    os.path.expanduser("~/PaddleOCR/doc/fonts/simfang.ttf"),
    os.path.expanduser("~/PP-OCRv6/doc/fonts/simfang.ttf"),
    "/opt/paddleocr/doc/fonts/simfang.ttf",
    "/usr/share/fonts/truetype/paddleocr/simfang.ttf",
    "/usr/local/share/fonts/simfang.ttf",
    os.path.expanduser("~/.local/share/fonts/simfang.ttf"),
    os.path.expanduser("~/.fonts/simfang.ttf"),
    "C:\\Windows\\Fonts\\simsun.ttc",
    "C:\\Windows\\Fonts\\simfang.ttf",
    "/System/Library/Fonts/Songti.ttc",
    "/System/Library/Fonts/STSong.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_CMAP_CACHE: dict[str, set[int] | None] = {}


def _load_cmap(font_path: str) -> set[int] | None:
    """Code points covered by ``font_path``, or None if unknown / unavailable."""
    if font_path in _CMAP_CACHE:
        return _CMAP_CACHE[font_path]
    try:
        from fontTools.ttLib import TTFont, TTCollection
    except Exception:
        _CMAP_CACHE[font_path] = None
        return None
    try:
        if font_path.lower().endswith(".ttc"):
            ttfont = TTCollection(font_path, lazy=True).fonts[0]
        else:
            ttfont = TTFont(font_path, lazy=True)
        cmap = ttfont.getBestCmap() or {}
        _CMAP_CACHE[font_path] = set(cmap.keys())
    except Exception:
        _CMAP_CACHE[font_path] = None
    return _CMAP_CACHE[font_path]


def _font_missing_chars(font_path: str, texts: list[str]) -> list[str]:
    """Characters in ``texts`` that ``font_path`` cannot render."""
    cmap = _load_cmap(font_path)
    if cmap is None:
        return []
    missing = set()
    for t in texts:
        if not t:
            continue
        for ch in t:
            if ord(ch) not in cmap:
                missing.add(ch)
    return sorted(missing)


def parse_log(text: str) -> list[dict]:
    results = []
    for m in _BOX_RE.finditer(text):
        pts = [
            (int(m.group("x0")), int(m.group("y0"))),
            (int(m.group("x1")), int(m.group("y1"))),
            (int(m.group("x2")), int(m.group("y2"))),
            (int(m.group("x3")), int(m.group("y3"))),
        ]
        results.append(
            {
                "text": m.group("text"),
                "score": float(m.group("score")),
                "box": pts,
                "det_score": float(m.group("det_score")),
            }
        )
    return results


def find_default_font(required_texts: list[str] | None = None) -> str | None:
    """First existing candidate that covers the OCR output."""
    fallback = None
    for candidate in _FONT_CANDIDATES:
        if not os.path.exists(candidate):
            continue
        if fallback is None:
            fallback = candidate
        # If fontTools is unavailable, coverage is unknown; retain the
        # original candidate order as the fallback behavior.
        if not required_texts or not _font_missing_chars(candidate, required_texts):
            return candidate
    return fallback


# Drawing helpers adapted from PaddleOCR tools/infer/utility.py.


def order_points_clockwise(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reorder 4 points to [TL, TR, BR, BL] (PaddleOCR order_points_clockwise)."""
    arr = np.array(pts, dtype=np.float32)
    s = arr.sum(axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = arr[np.argmin(s)]
    rect[2] = arr[np.argmax(s)]
    tmp = np.delete(arr, (int(np.argmin(s)), int(np.argmax(s))), axis=0)
    diff = np.diff(tmp, axis=1)
    rect[1] = tmp[int(np.argmin(diff))]
    rect[3] = tmp[int(np.argmax(diff))]
    return [(int(p[0]), int(p[1])) for p in rect]


def create_font(txt: str, sz: tuple[int, int], font_path: str) -> ImageFont.ImageFont:
    """Choose a font size that fits ``txt`` inside a ``sz = (width, height)`` box."""
    font_size = max(1, int(sz[1] * 0.99))
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    if int(PIL.__version__.split(".")[0]) < 10:
        length = font.getsize(txt)[0]  # type: ignore[attr-defined]
    else:
        length = font.getlength(txt)
    if length > sz[0] and length > 0:
        font_size = max(1, int(font_size * sz[0] / length))
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def _draw_topleft(
    draw: ImageDraw.ImageDraw,
    txt: str,
    font: ImageFont.ImageFont,
    xy: tuple[float, float] = (0, 0),
) -> None:
    """Draw ``txt`` with visual top-left at ``xy`` (anchor=lt; getbbox fallback)."""
    try:
        draw.text(xy, txt, fill=(0, 0, 0), font=font, anchor="lt")
        return
    except (TypeError, ValueError):
        pass
    try:
        l, t, _, _ = font.getbbox(txt)
        draw.text((xy[0] - l, xy[1] - t), txt, fill=(0, 0, 0), font=font)
    except AttributeError:
        draw.text(xy, txt, fill=(0, 0, 0), font=font)


def _create_font_upright_cjk(
    txt: str,
    box_width: int,
    box_height: int,
    font_path: str,
) -> ImageFont.ImageFont:
    """Font size so each glyph fits ``box_width`` and the stack fits ``box_height``."""
    n = max(1, len(txt))
    per_h = box_height / n
    font_size = max(1, int(min(box_width, per_h) * 0.95))
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def draw_box_txt_fine(
    img_size: tuple[int, int],
    box: list[tuple[int, int]],
    txt: str,
    font_path: str,
) -> np.ndarray:
    """Warp ``txt`` into ``box`` (PaddleOCR draw_box_txt_fine + upright vertical CJK)."""
    box_arr = np.array(box, dtype=np.float32)
    box_height = int(
        math.hypot(box_arr[0, 0] - box_arr[3, 0], box_arr[0, 1] - box_arr[3, 1])
    )
    box_width = int(
        math.hypot(box_arr[0, 0] - box_arr[1, 0], box_arr[0, 1] - box_arr[1, 1])
    )
    box_height = max(box_height, 1)
    box_width = max(box_width, 1)

    # Vertical if AABB is tall enough and text has >=2 glyphs (looser than
    # upstream box_height > 2*box_width, which misses many esp-dl quads).
    aabb_w = float(box_arr[:, 0].max() - box_arr[:, 0].min())
    aabb_h = float(box_arr[:, 1].max() - box_arr[:, 1].min())
    n_visible = sum(1 for c in (txt or "") if not c.isspace())
    is_vertical = aabb_h >= 1.3 * max(aabb_w, 1.0) and aabb_h >= 20 and n_visible >= 2
    if is_vertical:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = _create_font_upright_cjk(txt, box_width, box_height, font_path)
            n = len(txt)
            step = box_height / n
            for i, ch in enumerate(txt):
                try:
                    l, t, r, b = font.getbbox(ch)
                    gw = r - l
                    gh = b - t
                except AttributeError:
                    gw = gh = 0
                cx = box_width / 2 - gw / 2 - (l if gw else 0)
                cy = i * step + step / 2 - gh / 2 - (t if gh else 0)
                try:
                    draw_text.text(
                        (box_width / 2, i * step + step / 2),
                        ch,
                        fill=(0, 0, 0),
                        font=font,
                        anchor="mm",
                    )
                except (TypeError, ValueError):
                    draw_text.text((cx, cy), ch, fill=(0, 0, 0), font=font)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            _draw_topleft(draw_text, txt, font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(
        np.array(img_text, dtype=np.uint8),
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped


def draw_ocr_box_txt(
    image: Image.Image,
    boxes: list[list[tuple[int, int]]],
    txts: list[str],
    scores: list[float],
    drop_score: float,
    font_path: str,
    seed: int = 0,
) -> np.ndarray:
    """Two-column layout (PaddleOCR draw_ocr_box_txt; RNG seeded)."""
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(seed)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        box = order_points_clockwise(box)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon([tuple(p) for p in box], fill=color)
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Source image (the one embedded in the firmware).",
    )
    ap.add_argument(
        "--log",
        default="-",
        help='Serial log file to parse. "-" means stdin (default).',
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated image path. Default: <image>_annotated.jpg.",
    )
    ap.add_argument(
        "--font",
        default=None,
        help="TTF/TTC font path. Use a CJK font for Chinese text.",
    )
    ap.add_argument(
        "--drop-score",
        type=float,
        default=0.0,
        help="Extra rec-score filter for drawing (default 0.0; device already applies drop_score).",
    )
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Only print parsed results, do not render an image.",
    )
    args = ap.parse_args()

    if args.log == "-":
        log_text = sys.stdin.read()
    else:
        log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")

    results = parse_log(log_text)
    if not results:
        print("No OCR result lines matched. Expected lines like:")
        print('  text="...", score=0.xx, box=[x,y x,y x,y x,y], det_score=0.xx')
        sys.exit(1)

    print(f"[i] parsed {len(results)} OCR line(s):")
    for i, r in enumerate(results):
        pts = " ".join(f"({x},{y})" for x, y in r["box"])
        print(
            f"  #{i:02d} det={r['det_score']:.3f} rec={r['score']:.3f}  {pts}  text={r['text']!r}"
        )

    if args.print_only:
        return

    if not args.image.exists():
        print(f"[!] image not found: {args.image}", file=sys.stderr)
        sys.exit(2)

    boxes = [r["box"] for r in results]
    txts = [r["text"] for r in results]
    scores = [r["score"] for r in results]

    font_path = args.font or find_default_font(txts)
    if font_path is None:
        print(
            "[!] no default font found; pass --font <TTF/TTC> for Chinese glyphs.",
            file=sys.stderr,
        )
        sys.exit(3)
    print(f"[i] using font: {font_path}")
    missing = _font_missing_chars(font_path, txts)
    if missing:
        print(
            f"[!] font {font_path!r} is missing glyphs for: {''.join(missing)!r} "
            f"(they will render as empty boxes). simfang.ttf ships with a "
            f"trimmed cmap; pass --font <TTF/TTC> to switch to a font with "
            f"wider Unicode coverage (e.g. NotoSerifCJK-Regular.ttc, "
            f"NotoSansCJK-Regular.ttc, or wqy-microhei.ttc).",
            file=sys.stderr,
        )

    image = Image.open(args.image).convert("RGB")

    canvas = draw_ocr_box_txt(
        image=image,
        boxes=boxes,
        txts=txts,
        scores=scores,
        drop_score=args.drop_score,
        font_path=font_path,
    )

    output_path = args.output or args.image.with_name(
        args.image.stem + "_annotated.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)
    print(f"[OK] wrote {output_path}  ({len(results)} boxes)")


if __name__ == "__main__":
    main()
