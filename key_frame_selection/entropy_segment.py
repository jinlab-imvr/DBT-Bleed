"""
Keyframe Selection via Low-Level Visual Features.

Strategy:
  - entropy:  Red channel Shannon entropy + hierarchical segment elimination.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Per-frame scoring functions
# ---------------------------------------------------------------------------

def compute_red_entropy(image_bgr: np.ndarray) -> float:
    """Shannon entropy of the Red channel histogram (256 bins)."""
    red = image_bgr[:, :, 2]  # BGR -> R is channel 2
    hist = cv2.calcHist([red], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))



# ---------------------------------------------------------------------------
# Score all frames in a clip
# ---------------------------------------------------------------------------

def compute_all_scores(
    frame_dir: Path,
    start_idx: int,
    end_idx: int,
    pad: int,
    ext: str,
    prefix: str,
    method: str = "entropy",
    resize: int = 64,
) -> np.ndarray:
    """Read all N frames at low resolution and return a 1-D score array.

    Parameters
    ----------
    frame_dir : directory containing frame images.
    start_idx, end_idx : inclusive frame index range.
    pad : zero-padding width for frame filenames.
    ext : file extension (e.g. ".jpg").
    prefix : filename prefix (e.g. "BBP02_").
    method : "entropy" (Red channel Shannon entropy).
    resize : resize dimension for fast scoring (default 64x64).

    Returns
    -------
    np.ndarray of shape (end_idx - start_idx + 1,) with per-frame scores.
    """
    total = end_idx - start_idx + 1
    scores = np.zeros(total, dtype=np.float64)

    for i, idx in enumerate(range(start_idx, end_idx + 1)):
        fname = f"{prefix}{idx:0{pad}d}{ext}"
        fpath = frame_dir / fname
        img = cv2.imread(str(fpath))

        if img is None:
            scores[i] = 0.0
            continue

        img = cv2.resize(img, (resize, resize))

        if method == "entropy":
            scores[i] = compute_red_entropy(img)
        else:
            raise ValueError(f"Unknown scoring method: {method!r}")

    return scores


# ---------------------------------------------------------------------------
# Segment utilities
# ---------------------------------------------------------------------------

def _split_into_segments(items: List[int], n_segments: int) -> List[List[int]]:
    """Divide a list into n roughly-equal consecutive groups."""
    n = len(items)
    if n_segments <= 0:
        return [items]
    if n_segments >= n:
        return [[x] for x in items]

    segments = []
    edges = np.linspace(0, n, n_segments + 1)
    for k in range(n_segments):
        lo = int(np.floor(edges[k]))
        hi = int(np.floor(edges[k + 1]))
        if hi <= lo:
            hi = lo + 1
        segments.append(items[lo:hi])
    return segments


# ---------------------------------------------------------------------------
# Hierarchical segment selection
# ---------------------------------------------------------------------------

def entropy_segment_select(
    scores: np.ndarray,
    start_idx: int,
    end_idx: int,
    num_frames: int,
    segment_size: int = 4,
) -> List[int]:
    """Select exactly `num_frames` indices via hierarchical segment elimination.

    Algorithm:
      1. Start with all frame indices and their scores.
      2. Divide into segments of ~segment_size, pick the highest-scoring
         frame per segment.
      3. Repeat until exactly `num_frames` remain.
      4. If a round would reduce below `num_frames`, divide into exactly
         `num_frames` segments instead.

    Returns sorted list of absolute frame indices.
    """
    total = end_idx - start_idx + 1

    # Edge case: not enough frames
    if total <= num_frames:
        offsets = np.linspace(0, total - 1, num_frames)
        return (start_idx + np.round(offsets).astype(int)).tolist()

    # Build initial list of (absolute_index, score) pairs
    candidates = list(range(start_idx, end_idx + 1))

    while len(candidates) > num_frames:
        n_segs = max(1, len(candidates) // segment_size)

        # If this round would reduce below num_frames, use exactly num_frames segments
        if n_segs < num_frames:
            n_segs = num_frames
        # Clamp to current count
        if n_segs >= len(candidates):
            n_segs = num_frames

        segments = _split_into_segments(candidates, n_segs)
        new_candidates = []
        for seg in segments:
            # Pick the highest-scoring frame in this segment
            best_idx = max(seg, key=lambda idx: scores[idx - start_idx])
            new_candidates.append(best_idx)
        candidates = new_candidates

    return sorted(candidates[:num_frames])


def entropy_segment_select_jitter(
    scores: np.ndarray,
    start_idx: int,
    end_idx: int,
    num_frames: int,
    segment_size: int = 4,
    jitter_prob: float = 0.3,
) -> List[int]:
    """Training variant: with probability `jitter_prob`, pick the 2nd-best
    frame instead of the best per segment.

    Falls back to best when segment has only 1 frame.
    """
    total = end_idx - start_idx + 1

    if total <= num_frames:
        offsets = np.linspace(0, total - 1, num_frames)
        return (start_idx + np.round(offsets).astype(int)).tolist()

    candidates = list(range(start_idx, end_idx + 1))

    while len(candidates) > num_frames:
        n_segs = max(1, len(candidates) // segment_size)

        if n_segs < num_frames:
            n_segs = num_frames
        if n_segs >= len(candidates):
            n_segs = num_frames

        segments = _split_into_segments(candidates, n_segs)
        new_candidates = []
        for seg in segments:
            sorted_seg = sorted(seg, key=lambda idx: scores[idx - start_idx], reverse=True)
            if len(sorted_seg) >= 2 and np.random.random() < jitter_prob:
                new_candidates.append(sorted_seg[1])
            else:
                new_candidates.append(sorted_seg[0])
        candidates = new_candidates

    return sorted(candidates[:num_frames])


# ---------------------------------------------------------------------------
# Top-level convenience wrapper
# ---------------------------------------------------------------------------

def select_keyframes(
    frame_dir: Path,
    start_idx: int,
    end_idx: int,
    num_frames: int,
    prefix: str = "",
    pad: int = 8,
    ext: str = ".jpg",
    method: str = "entropy",
    segment_size: int = 4,
    resize: int = 64,
    jitter: bool = False,
    jitter_prob: float = 0.3,
) -> List[int]:
    """Score frames and select keyframes in one call."""
    scores = compute_all_scores(
        Path(frame_dir), start_idx, end_idx, pad, ext, prefix, method, resize,
    )
    if jitter:
        return entropy_segment_select_jitter(
            scores, start_idx, end_idx, num_frames, segment_size, jitter_prob,
        )
    return entropy_segment_select(scores, start_idx, end_idx, num_frames, segment_size)
