"""Build paired quality/attack comparisons from preserved evaluation artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "comparison_completed_expanded"
THRESHOLDS = (0.90, 0.85, 0.80, 0.70, 0.50)
SOURCES = {
    "No attack": ROOT / "outputs/baselines/none/results.csv",
    "FGSM": ROOT / "outputs/baselines/fgsm/results.csv",
    "PGD": ROOT / "outputs/baselines/pgd/results.csv",
    "Frequency low-pass": ROOT / "outputs/baselines/frequency/results.csv",
    "Basic PPO": ROOT / "outputs/basic_actions_seed42_evaluation/results.csv",
    "Expanded PPO": ROOT / "outputs/expanded_actions_seed42_evaluation_completed/results.csv",
    "Quality PPO": ROOT / "outputs/expanded_quality_seed42_evaluation_completed/results.csv",
}
RUNTIMES = {
    "No attack": 2.5852177499909885,
    "FGSM": 66.0675179589889,
    "PGD": 555.9422064579558,
    "Frequency low-pass": 5.1531594170373864,
    "Basic PPO": 9.32,
    "Expanded PPO": 480.0880947089754,
    "Quality PPO": 487.45449945901055,
}


def _svg_chart(path: Path, title: str, xlabel: str, ylabel: str,
               series: list[tuple[str, list[tuple[float, float]]]],
               x_ticks: list[tuple[float, str]] | None = None) -> None:
    """Write a dependency-free SVG line/scatter chart."""
    width, height, left, top, right, bottom = 900, 520, 80, 50, 30, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    all_x = [x for _, points in series for x, _ in points]
    if x_ticks:
        all_x.extend(x for x, _ in x_ticks)
    x_min, x_max = min(all_x), max(all_x)
    if x_min == x_max: x_max += 1
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    sx = lambda x: left + (x - x_min) / (x_max - x_min) * plot_w
    sy = lambda y: top + (100 - y) / 100 * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>']
    for y in range(0, 101, 20):
        parts += [f'<line x1="{left}" y1="{sy(y):.1f}" x2="{left+plot_w}" y2="{sy(y):.1f}" stroke="#ddd"/>',
                  f'<text x="{left-10}" y="{sy(y)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y}</text>']
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>',
              f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>']
    ticks = x_ticks or [(x_min, f"{x_min:.2f}"), (x_max, f"{x_max:.2f}")]
    for x, label in ticks:
        parts.append(f'<text x="{sx(x):.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>')
    for index, (name, points) in enumerate(series):
        color = colors[index % len(colors)]
        if len(points) > 1:
            coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}"/>')
        lx, ly = points[-1]
        parts.append(f'<text x="{sx(lx)+7:.1f}" y="{sy(ly)-7:.1f}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>')
    parts += [f'<text x="{left+plot_w/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="14">{xlabel}</text>',
              f'<text transform="translate(18 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>', '</svg>']
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    if "eligible" in frame:
        return frame[frame["eligible"].astype(bool)].copy()
    return frame[(frame.ground_truth_label == "FAKE") & (frame.initial_label == "FAKE")].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {name: eligible(pd.read_csv(path)) for name, path in SOURCES.items()}
    expected = set(frames["Expanded PPO"].image_path)
    if len(expected) != 16 or any(set(frame.image_path) != expected for frame in frames.values()):
        raise RuntimeError("Methods do not contain the same 16 eligible images")

    rows = []
    paired = pd.DataFrame({"image_path": sorted(expected)})
    for name, frame in frames.items():
        frame = frame.set_index("image_path").loc[paired.image_path]
        successes = int(frame.attack_success.astype(bool).sum())
        low, high = wilson(successes, len(frame))
        row = {
            "method": name,
            "successes": successes,
            "eligible": len(frame),
            "raw_asr": successes / len(frame),
            "asr_wilson_95_low": low,
            "asr_wilson_95_high": high,
            "mean_ssim": float(frame.ssim.mean()),
            "median_ssim": float(frame.ssim.median()),
            "std_ssim": float(frame.ssim.std(ddof=0)),
            "mean_psnr": None if math.isinf(float(frame.psnr.mean())) else float(frame.psnr.mean()),
            "runtime_seconds": RUNTIMES[name],
            "seconds_per_image": RUNTIMES[name] / 40,
        }
        for threshold in THRESHOLDS:
            count = int((frame.attack_success.astype(bool) & (frame.ssim >= threshold)).sum())
            row[f"successes_ssim_gte_{threshold:.2f}"] = count
            row[f"asr_ssim_gte_{threshold:.2f}"] = count / len(frame)
        rows.append(row)
        key = name.lower().replace(" ", "_").replace("-", "_")
        paired[f"{key}_success"] = frame.attack_success.astype(bool).to_numpy()
        paired[f"{key}_ssim"] = frame.ssim.to_numpy()
        paired[f"{key}_psnr"] = frame.psnr.to_numpy()
        if "final_fake_probability" in frame:
            paired[f"{key}_final_fake_probability"] = frame.final_fake_probability.to_numpy()

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    paired.to_csv(OUT / "paired_eligible_results.csv", index=False)
    (OUT / "comparison.json").write_text(
        json.dumps(comparison.where(pd.notna(comparison), None).to_dict("records"), indent=2) + "\n"
    )

    plot = comparison[comparison.method != "No attack"]
    _svg_chart(
        OUT / "asr_vs_ssim.svg", "Raw ASR vs mean SSIM", "Mean SSIM", "Raw ASR (%)",
        [(row.method, [(float(row.mean_ssim), 100 * float(row.raw_asr))]) for _, row in plot.iterrows()],
        [(0.3, "0.30"), (0.5, "0.50"), (0.7, "0.70"), (0.9, "0.90"), (1.0, "1.00")],
    )
    x_values = list(range(len(THRESHOLDS)))
    _svg_chart(
        OUT / "constrained_asr.svg", "Quality-constrained ASR", "Minimum SSIM", "ASR (%)",
        [(row.method, [(x, 100 * float(row[f"asr_ssim_gte_{t:.2f}"])) for x, t in zip(x_values, THRESHOLDS)]) for _, row in plot.iterrows()],
        [(x, f"≥ {t:.2f}") for x, t in zip(x_values, THRESHOLDS)],
    )


if __name__ == "__main__":
    main()
