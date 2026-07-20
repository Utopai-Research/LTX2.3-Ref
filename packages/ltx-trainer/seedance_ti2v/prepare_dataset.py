#!/usr/bin/env python3
"""
Prepare the Seedance dataset for LTX-2.3 first-frame TI2V fine-tuning.

What it does
------------
1. Reads the source ``dataset.jsonl`` (one record per generated clip).
2. Keeps only records whose aspect ratio matches ``--aspect-ratio`` (default 16:9)
   AND whose ``video`` file actually exists on disk.
3. Deterministically (seeded) shuffles and splits into a TEST hold-out and a TRAIN set.
4. Writes ``train_ti2v.jsonl`` in the format the LTX trainer expects:
       {"caption": <raw prompt>, "media_path": "videos/<id>.mp4", "job_id": <id>}
   The trainer / preprocessor resolve ``media_path`` relative to the jsonl's parent dir,
   so the jsonl is written INTO the dataset root.
5. Builds a self-contained TEST set under ``<dataset_root>/test_set/``:
       test_set/videos/<id>.mp4          (copy of the held-out clip)
       test_set/first_frames/<id>.png    (frame 0 — the I2V conditioning image)
       test_set/test.jsonl               ({job_id, caption, first_frame, video, duration_s})

Why no first frames for the TRAIN set?
--------------------------------------
The LTX trainer performs first-frame image conditioning IMPLICITLY: with probability
``first_frame_conditioning_p`` it keeps the clean first latent frame and denoises the rest.
That clean first frame IS "the image" (== first frame of the video). So training needs only
(video, caption); no separate image files. Real first-frame PNGs are only needed for the
held-out TEST set so you can run I2V inference later.

This script intentionally depends only on PyAV + stdlib (no torch), so it runs fast with the
project venv python without spinning up the full training stack.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_first_frame(video_path: Path, out_png: Path) -> bool:
    """Decode frame 0 of the video and save it as a PNG. Returns True on success."""
    import av  # local import so --help works without av installed

    try:
        container = av.open(str(video_path))
        for frame in container.decode(video=0):
            frame.to_image().save(str(out_png))
            container.close()
            return True
        container.close()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] first-frame extraction failed for {video_path.name}: {e}", file=sys.stderr)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-jsonl", default="/home/zheng/seedance_video_v0/dataset.jsonl",
                   help="Source dataset.jsonl")
    p.add_argument("--aspect-ratio", default="16:9", help="Aspect ratio to keep (e.g. 16:9). Use 'all' to keep every AR.")
    p.add_argument("--num-test", type=int, default=50, help="Number of clips to hold out for testing")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for a reproducible split")
    p.add_argument("--caption-field", default="prompt", help="Source field used as the caption (raw prompt)")
    p.add_argument("--video-field", default="video", help="Source field holding the relative video path")
    p.add_argument("--train-name", default="train_ti2v.jsonl", help="Output train jsonl filename (written in dataset root)")
    p.add_argument("--max-seconds", type=int, default=0,
                   help="If > 0, drop clips longer than this many seconds (0 = keep all)")
    p.add_argument("--no-copy-test-videos", action="store_true",
                   help="Do not copy test videos into test_set/videos (still extracts first frames + writes test.jsonl)")
    args = p.parse_args()

    jsonl_path = Path(args.dataset_jsonl).resolve()
    data_root = jsonl_path.parent
    records = load_records(jsonl_path)
    print(f"Loaded {len(records):,} records from {jsonl_path}")

    # --- filter: aspect ratio + present video file (+ optional max length) ---
    kept = []
    n_ar_drop = n_missing = n_toolong = 0
    for r in records:
        if args.aspect_ratio != "all" and r.get("aspect_ratio") != args.aspect_ratio:
            n_ar_drop += 1
            continue
        rel_video = r.get(args.video_field)
        if not rel_video:
            n_missing += 1
            continue
        video_path = data_root / rel_video
        if not video_path.is_file():
            n_missing += 1
            continue
        if args.max_seconds and (r.get("total_duration_s") or 0) > args.max_seconds:
            n_toolong += 1
            continue
        kept.append(r)

    print(f"Filter: dropped {n_ar_drop:,} (aspect != {args.aspect_ratio}), "
          f"{n_missing:,} (missing/blank video), {n_toolong:,} (> {args.max_seconds}s)")
    print(f"Usable clips: {len(kept):,}")

    if args.num_test >= len(kept):
        raise SystemExit(f"--num-test ({args.num_test}) must be < usable clips ({len(kept)})")

    # --- deterministic split: sort by job_id for stability, then seeded shuffle ---
    kept.sort(key=lambda r: r.get("job_id", r.get(args.video_field, "")))
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    test_records = kept[: args.num_test]
    train_records = kept[args.num_test :]
    print(f"Split (seed={args.seed}): {len(train_records):,} train / {len(test_records):,} test")

    # --- write train jsonl (trainer format) into the dataset root ---
    train_path = data_root / args.train_name
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps({
                "caption": r[args.caption_field],
                "media_path": r[args.video_field],
                "job_id": r.get("job_id", ""),
            }, ensure_ascii=False) + "\n")
    print(f"Wrote train jsonl -> {train_path}  ({len(train_records):,} lines)")

    # --- build the test set ---
    test_dir = data_root / "test_set"
    ff_dir = test_dir / "first_frames"
    tv_dir = test_dir / "videos"
    ff_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_test_videos:
        tv_dir.mkdir(parents=True, exist_ok=True)

    test_jsonl = test_dir / "test.jsonl"
    n_frames_ok = 0
    with test_jsonl.open("w", encoding="utf-8") as f:
        for r in test_records:
            jid = r.get("job_id", "")
            src_video = data_root / r[args.video_field]
            ff_png = ff_dir / f"{jid}.png"
            ok = extract_first_frame(src_video, ff_png)
            n_frames_ok += int(ok)

            dst_video = src_video
            if not args.no_copy_test_videos:
                dst_video = tv_dir / f"{jid}.mp4"
                if not dst_video.exists():
                    shutil.copy2(src_video, dst_video)

            f.write(json.dumps({
                "job_id": jid,
                "caption": r[args.caption_field],
                "first_frame": str(ff_png),
                "video": str(dst_video),
                "duration_s": r.get("total_duration_s"),
            }, ensure_ascii=False) + "\n")

    print(f"Test set -> {test_dir}")
    print(f"  first frames extracted: {n_frames_ok}/{len(test_records)} -> {ff_dir}")
    if not args.no_copy_test_videos:
        print(f"  test videos copied      -> {tv_dir}")
    print(f"  test manifest           -> {test_jsonl}")
    print("\nDone. Next: run preprocessing on the train jsonl, then train.")


if __name__ == "__main__":
    main()
