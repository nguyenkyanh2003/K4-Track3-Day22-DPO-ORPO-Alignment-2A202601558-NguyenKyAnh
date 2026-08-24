#!/usr/bin/env python3
"""Render reproducible submission evidence from the saved NB4 judge JSON."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "data" / "eval" / "judge_results.json"
TARGET = REPO / "submission" / "screenshots" / "05-judge-output.png"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows_fonts = Path("C:/Windows/Fonts")
    candidates = [
        windows_fonts / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    selected = [by_id[index] for index in (4, 5, 8)]

    canvas = Image.new("RGB", (1800, 1120), "#f7f5ef")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(42, bold=True)
    subtitle_font = load_font(26)
    heading_font = load_font(29, bold=True)
    body_font = load_font(25)
    footer_font = load_font(23)

    draw.text((70, 48), "NB4 · API judge verdicts", font=title_font, fill="#172238")
    draw.text(
        (70, 105),
        "Verbatim from data/eval/judge_results.json · Nguyễn Kỳ Anh · 2A202601558",
        font=subtitle_font,
        fill="#555b66",
    )

    card_y = 170
    for row in selected:
        draw.rounded_rectangle(
            (65, card_y, 1735, card_y + 245),
            radius=18,
            fill="#ffffff",
            outline="#d7d2c8",
            width=3,
        )
        draw.text(
            (100, card_y + 28),
            f"Prompt #{row['id']} · {row['category']} · verdict: {row['winner']}",
            font=heading_font,
            fill="#223047",
        )
        wrapped = textwrap.fill(row["justification"], width=110)
        draw.multiline_text(
            (100, card_y + 88),
            wrapped,
            font=body_font,
            fill="#30343b",
            spacing=11,
        )
        card_y += 275

    draw.text(
        (900, 1055),
        "8 prompts · 4 helpfulness + 4 safety · B wins 2 · ties 6 · A wins 0",
        font=footer_font,
        fill="#555b66",
        anchor="mm",
    )
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(TARGET, format="PNG", optimize=True)
    print(f"Wrote {TARGET.relative_to(REPO)}")


if __name__ == "__main__":
    main()
