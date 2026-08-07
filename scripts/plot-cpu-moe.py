#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]


def chart(rows, key, title, ylabel, output):
    width, height = 820, 460
    left, right, top, bottom = 80, 25, 50, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    contexts = sorted({row["context"] for row in rows})
    maximum = max(row[key] for row in rows) * 1.15

    def x(value):
        return left + value / 40 * plot_w

    def y(value):
        return top + (1 - value / maximum) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{title}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        py = y(value)
        svg += [
            f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#e5e7eb"/>',
            f'<text x="{left-10}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>',
        ]
    for value in range(0, 41, 8):
        px = x(value)
        svg += [
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#f3f4f6"/>',
            f'<text x="{px:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{value}</text>',
        ]
    svg += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<text x="{left+plot_w/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="14">CPU MoE layers</text>',
        f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 18 {top+plot_h/2})" font-family="sans-serif" font-size="14">{ylabel}</text>',
    ]
    for index, context in enumerate(contexts):
        color = COLORS[index]
        points = sorted((row for row in rows if row["context"] == context), key=lambda row: row["n_cpu_moe"])
        coordinates = " ".join(f'{x(row["n_cpu_moe"]):.1f},{y(row[key]):.1f}' for row in points)
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in points:
            svg.append(f'<circle cx="{x(row["n_cpu_moe"]):.1f}" cy="{y(row[key]):.1f}" r="4" fill="{color}"/>')
        label = f'{context // 1024}k context'
        ly = top + index * 22
        svg += [
            f'<line x1="{width-175}" y1="{ly}" x2="{width-145}" y2="{ly}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{width-138}" y="{ly+4}" font-family="sans-serif" font-size="12">{label}</text>',
        ]
    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n")


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "results/cpu-moe-results.json")
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "results/plots")
    rows = json.loads(source.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("generation_tps_mean", "CPU MoE offload: generation speed", "cpu_moe_generation"),
        ("prompt_tps_mean", "CPU MoE offload: prompt processing", "cpu_moe_prompt"),
    ]
    for key, title, name in charts:
        svg, jpeg, png = (output_dir / f"{name}.{suffix}" for suffix in ("svg", "jpg", "png"))
        chart(rows, key, title, "tokens / second", svg)
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "100", svg, "--out", jpeg], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["sips", "-s", "format", "png", jpeg, "--out", png], check=True, stdout=subprocess.DEVNULL)
        svg.unlink()
        jpeg.unlink()
    assert (output_dir / "cpu_moe_generation.png").stat().st_size > 1000


if __name__ == "__main__":
    main()
