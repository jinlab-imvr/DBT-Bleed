import csv
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VideoCSVDataset(Dataset):
    """
    Dataset for videos stored as frame folders with metadata in a CSV file.

    CSV columns (required):
      - clip_path: path to directory containing frames
      - num_frames: total frames in the video (informational)
      - gt: 0 for normal, 1 for abnormal
      - start_frame: filename of first frame (e.g., 000001.jpg)
      - end_frame: filename of last frame (e.g., 000240.jpg)

    Each sample returns:
      - video tensor [T, 3, H, W]
      - label tensor scalar (float32)
    """

    def __init__(
        self,
        csv_path: str,
        num_frames: int,
        resize: int = 240,
        sample_mode: str = "uniform",
        is_train: bool = False,
        return_path: bool = False,
        entropy_method: str = "entropy",
        segment_size: int = 4,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames}")

        self.csv_dir = self.csv_path.parent
        self.num_frames = num_frames
        self.sample_mode = sample_mode
        self.is_train = is_train
        self.return_path = return_path
        self.entropy_method = entropy_method
        self.segment_size = segment_size

        self.entries = self._read_csv(self.csv_path)

        self.transform = transforms.Compose([
            transforms.Resize((resize, resize), Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        frame_dir = entry["frame_dir"]
        prefix = entry["prefix"]
        start_idx = entry["start_idx"]
        end_idx = entry["end_idx"]
        pad = entry["pad"]
        ext = entry["ext"]

        frame_indices = self._sample_indices(
            start_idx, end_idx, frame_dir=str(frame_dir),
            prefix=prefix, pad=pad, ext=ext,
        )

        frames = []
        for frame_idx in frame_indices:
            frame_path = self._frame_path(frame_dir, frame_idx, pad, ext, prefix)
            if not frame_path.exists():
                raise FileNotFoundError(f"Missing frame: {frame_path}")
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                frames.append(self.transform(img))

        video = torch.stack(frames, dim=0)  # [T, 3, H, W]
        label = torch.tensor(entry["gt"], dtype=torch.float32)
        if self.return_path:
            return video, label, str(entry["clip_path"])
        return video, label

    def _read_csv(self, csv_path: Path):
        entries = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"clip_path", "num_frames", "gt", "start_frame", "end_frame"}
            if not required.issubset(reader.fieldnames or []):
                missing = required - set(reader.fieldnames or [])
                raise ValueError(f"CSV missing columns: {sorted(missing)}")

            for row in reader:
                clip_path = Path(row["clip_path"])
                if not clip_path.is_absolute():
                    clip_path = (self.csv_dir / clip_path).resolve()

                start_idx, start_pad, start_ext, start_prefix = self._parse_frame_name(row["start_frame"])
                end_idx, end_pad, end_ext, end_prefix = self._parse_frame_name(row["end_frame"])

                pad = max(start_pad, end_pad)
                ext = start_ext if start_ext else end_ext
                prefix = start_prefix              # e.g. "BBP02_"

                # clip_path is like .../BBP02/BBP02_00000001 (clip id, not a dir)
                # frames live in the parent directory (.../BBP02/)
                frame_dir = clip_path.parent

                entries.append({
                    "clip_path": clip_path,
                    "frame_dir": frame_dir,
                    "prefix": prefix,
                    "num_frames": int(row["num_frames"]),
                    "gt": int(row["gt"]),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "pad": pad,
                    "ext": ext,
                })

        return entries

    def _parse_frame_name(self, name: str):
        name = os.path.basename(str(name))
        stem, ext = os.path.splitext(name)
        match = re.search(r"(\d+)$", stem)
        if match is None:
            raise ValueError(f"Frame name has no numeric index: {name}")
        digits = match.group(1)
        prefix = stem[:match.start()]          # e.g. "BBP02_" from "BBP02_00000001"
        return int(digits), len(digits), ext, prefix

    def _infer_extension(self, clip_dir: Path):
        for p in sorted(clip_dir.iterdir()):
            if not p.is_file():
                continue
            if p.stem.isdigit():
                return p.suffix
        raise FileNotFoundError(f"No frame files found in: {clip_dir}")

    def _frame_path(self, frame_dir: Path, idx: int, pad: int, ext: str, prefix: str = ""):
        if ext == "":
            ext = self._infer_extension(frame_dir)
        return frame_dir / f"{prefix}{idx:0{pad}d}{ext}"

    def _sample_indices(self, start_idx: int, end_idx: int, frame_dir: str = "",
                        prefix: str = "", pad: int = 8, ext: str = ".jpg"):
        total = end_idx - start_idx + 1
        if total <= 0:
            raise ValueError(f"Invalid frame range: {start_idx}..{end_idx}")

        # Entropy-segment-based sampling modes
        if self.sample_mode in ("entropy_segment", "entropy_segment_jitter"):
            from key_frame_selection.entropy_segment import (
                compute_all_scores, entropy_segment_select, entropy_segment_select_jitter,
            )
            scores = compute_all_scores(
                Path(frame_dir), start_idx, end_idx, pad, ext, prefix, self.entropy_method,
            )
            if self.sample_mode == "entropy_segment_jitter" and self.is_train:
                return entropy_segment_select_jitter(
                    scores, start_idx, end_idx, self.num_frames, self.segment_size,
                )
            return entropy_segment_select(
                scores, start_idx, end_idx, self.num_frames, self.segment_size,
            )

        # Uniform sampling modes
        if self.sample_mode == "uniform_jitter" and self.is_train:
            edges = np.linspace(0, total, self.num_frames + 1)
            offsets = []
            for i in range(self.num_frames):
                seg_start = int(np.floor(edges[i]))
                seg_end = int(np.floor(edges[i + 1] - 1))
                if seg_end < seg_start:
                    seg_end = seg_start
                offsets.append(np.random.randint(seg_start, seg_end + 1))
            offsets = np.array(offsets, dtype=int)
        else:
            offsets = np.linspace(0, total - 1, self.num_frames)
            offsets = np.round(offsets).astype(int)

        return (start_idx + offsets).tolist()
