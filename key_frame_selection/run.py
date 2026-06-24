"""
Standalone CLI for keyframe selection.

Usage:
    python -m key_frame_selection.run \
        --csv_file dataset/mby140_N=300_Stride=200/val.csv \
        --output ./tmp/keyframes_entropy.json \
        --method entropy \
        --num_frames 16 --segment_size 4 \
        --video BBP02 --plot --verbose
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from key_frame_selection.entropy_segment import (
    compute_all_scores,
    entropy_segment_select,
)


def parse_frame_name(name: str):
    """Parse a frame filename into (index, pad, ext, prefix)."""
    name = os.path.basename(str(name))
    stem, ext = os.path.splitext(name)
    match = re.search(r"(\d+)$", stem)
    if match is None:
        raise ValueError(f"Frame name has no numeric index: {name}")
    digits = match.group(1)
    prefix = stem[: match.start()]
    return int(digits), len(digits), ext, prefix


# ---------------------------------------------------------------------------
# Thumbnail helper (shared by both plot functions)
# ---------------------------------------------------------------------------

def _make_thumbnail_canvas(selected_indices, frame_dir, prefix, pad, ext):
    import cv2

    n_selected = len(selected_indices)
    thumbs = []
    for idx in selected_indices:
        fname = f"{prefix}{idx:0{pad}d}{ext}"
        fpath = frame_dir / fname
        img = cv2.imread(str(fpath))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160, 120))
        else:
            img = np.zeros((120, 160, 3), dtype=np.uint8)
        thumbs.append(img)

    cols = min(8, n_selected)
    rows = (n_selected + cols - 1) // cols
    canvas_h = rows * 140
    canvas_w = cols * 175
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for t, (thumb, idx) in enumerate(zip(thumbs, selected_indices)):
        r, c = divmod(t, cols)
        y = r * 140 + 5
        x = c * 175 + 5
        canvas[y:y + 120, x:x + 160] = thumb
        cv2.putText(canvas, f"#{idx}", (x + 2, y + 115),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 50, 50), 1)

    return canvas


# ---------------------------------------------------------------------------
# Plot: entropy
# ---------------------------------------------------------------------------

def plot_clip(clip_name, all_indices, scores, start_idx, selected_indices,
              frame_dir, prefix, pad, ext, gt, out_dir, method_label="Score"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_selected = len(selected_indices)

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.25)

    label_str = "BLEEDING" if int(gt) == 1 else "NORMAL"
    label_color = "#d62728" if int(gt) == 1 else "#2ca02c"
    fig.suptitle(f"{clip_name}  [{label_str}]  (frames {start_idx}-{start_idx + len(scores) - 1})  method={method_label}",
                 fontsize=14, fontweight="bold", color=label_color)

    # --- Top-left: score curve ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(all_indices, scores, linewidth=0.8, color="#1f77b4", alpha=0.7, label=method_label)
    sel_offsets = [idx - start_idx for idx in selected_indices]
    sel_scores = scores[sel_offsets]
    ax1.scatter(selected_indices, sel_scores, color="#d62728", s=40, zorder=5,
                edgecolors="black", linewidths=0.5, label=f"Selected ({n_selected})")

    uniform_indices = np.linspace(start_idx, start_idx + len(scores) - 1, n_selected)
    uniform_indices = np.round(uniform_indices).astype(int)
    uni_offsets = np.clip(uniform_indices - start_idx, 0, len(scores) - 1)
    ax1.scatter(uniform_indices, scores[uni_offsets], color="#2ca02c", s=25, zorder=4,
                marker="^", alpha=0.6, label=f"Uniform ({n_selected})")

    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel(method_label)
    ax1.set_title(f"{method_label} Curve + Selected Keyframes")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Top-right: histogram ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(scores, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax2.axvline(scores.mean(), color="#d62728", linestyle="--", linewidth=1.5,
                label=f"Mean: {scores.mean():.3f}")
    ax2.axvline(np.median(scores), color="#ff7f0e", linestyle="--", linewidth=1.5,
                label=f"Median: {np.median(scores):.3f}")
    ax2.hist(sel_scores, bins=50, color="#d62728", alpha=0.5, edgecolor="black",
             linewidth=0.3, label="Selected")
    ax2.set_xlabel(method_label)
    ax2.set_ylabel("Count")
    ax2.set_title(f"{method_label} Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    stats_text = (
        f"N={len(scores)}  sel={n_selected}\n"
        f"min={scores.min():.3f}  max={scores.max():.3f}\n"
        f"std={scores.std():.3f}"
    )
    ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # --- Bottom: thumbnails ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title(f"Selected Keyframes ({n_selected} frames)", fontsize=11)
    ax3.axis("off")
    canvas = _make_thumbnail_canvas(selected_indices, frame_dir, prefix, pad, ext)
    ax3.imshow(canvas)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{clip_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path




# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Keyframe selection CLI (entropy)"
    )
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--method", type=str, default="entropy",
        choices=["entropy"],
        help="Scoring method: entropy (Red channel Shannon entropy)",
    )
    parser.add_argument("--num_frames", type=int, default=16, help="Number of keyframes")
    parser.add_argument("--segment_size", type=int, default=4,
                        help="Segment size for hierarchical elimination")
    parser.add_argument("--video", type=str, default="",
                        help="Process only clips matching this video name (e.g. 'BBP02')")
    parser.add_argument("--plot", action="store_true",
                        help="Save per-clip visualization PNGs")
    parser.add_argument("--plot_dir", type=str, default="",
                        help="Directory for plot PNGs (default: <output_dir>/plots)")
    parser.add_argument("--verbose", action="store_true", help="Print per-clip info")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    csv_dir = csv_path.parent

    results = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.video:
        rows = [r for r in rows if args.video in r["clip_path"]]
        print(f"Filtered to {len(rows)} clips matching video '{args.video}' from {csv_path}")
    else:
        print(f"Processing {len(rows)} clips from {csv_path}")

    if args.plot:
        plot_dir = args.plot_dir if args.plot_dir else str(Path(args.output).parent / "plots")
    else:
        plot_dir = ""

    for i, row in enumerate(rows):
        clip_path = Path(row["clip_path"])
        if not clip_path.is_absolute():
            clip_path = (csv_dir / clip_path).resolve()

        start_idx, start_pad, start_ext, start_prefix = parse_frame_name(row["start_frame"])
        end_idx, end_pad, end_ext, end_prefix = parse_frame_name(row["end_frame"])

        pad = max(start_pad, end_pad)
        ext = start_ext if start_ext else end_ext
        prefix = start_prefix
        frame_dir = clip_path.parent

        scores = compute_all_scores(
            frame_dir, start_idx, end_idx, pad, ext, prefix, args.method,
        )
        indices = entropy_segment_select(
            scores, start_idx, end_idx, args.num_frames, args.segment_size,
        )

        if args.verbose:
            total = end_idx - start_idx + 1
            print(
                f"  [{i+1}/{len(rows)}] {clip_path.name}: "
                f"{total} frames -> {len(indices)} keyframes "
                f"(score range: {scores.min():.3f}-{scores.max():.3f})"
            )

        if args.plot:
            all_indices = np.arange(start_idx, end_idx + 1)
            gt = row.get("gt", "0")
            save_path = plot_clip(
                clip_name=clip_path.name,
                all_indices=all_indices,
                scores=scores,
                start_idx=start_idx,
                selected_indices=indices,
                frame_dir=frame_dir,
                prefix=prefix,
                pad=pad,
                ext=ext,
                gt=gt,
                out_dir=plot_dir,
                method_label="Red Entropy",
            )
            if args.verbose:
                print(f"         -> plot saved to {save_path}")

        key = str(clip_path)
        results[key] = indices

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} clip keyframe selections to {out_path}")
    if args.plot:
        print(f"Plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
