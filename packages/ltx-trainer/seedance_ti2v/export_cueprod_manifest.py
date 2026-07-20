#!/usr/bin/env python3
"""
Export the seedance_cue_prod manifest.db -> dataset.jsonl for the r2v pipeline.

Produces the SAME schema as the old seedance_video_v0/dataset.jsonl so the existing
prepare_dataset.py / process_dataset.py / process_reference_images.py all work unchanged:
    {job_id, prompt, aspect_ratio, resolution, total_duration_s, video, ref_images[]}

Filter (the "16:9 + 720p/1080p" scope):
    - raw_metadata_json.aspect_ratio == 16:9
    - raw_metadata_json.resolution in {720p, 1080p}
    - raw_metadata_json.generate_audio == true   (AV training)
    - video file exists on disk
    - >= 1 reference image exists on disk (images/<job_id>/<n>.png)

ref_images are taken from the FILES ON DISK (images/<job_id>/*.png sorted by integer stem),
not the manifest 'images' list (whose entries are asset URIs, not local paths).

CPU-only (sqlite + stdlib). Run:
  uv run --no-sync python seedance_ti2v/export_cueprod_manifest.py \
    --root /home/zheng/seedance_cue_prod --out /home/zheng/seedance_cue_prod/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/home/zheng/seedance_cue_prod", help="cue_prod dataset root")
    ap.add_argument("--db", default="", help="manifest.db (default: <root>/manifest.db)")
    ap.add_argument("--out", default="", help="output dataset.jsonl (default: <root>/dataset.jsonl)")
    ap.add_argument("--aspect", default="16:9", help="keep only this aspect ratio")
    ap.add_argument("--resolutions", default="720p,1080p", help="comma list of resolutions to keep")
    ap.add_argument("--require-audio", action="store_true", default=True, help="require generate_audio")
    ap.add_argument("--max-images", type=int, default=9)
    args = ap.parse_args()

    root = Path(args.root)
    db = Path(args.db) if args.db else root / "manifest.db"
    out = Path(args.out) if args.out else root / "dataset.jsonl"
    keep_res = {r.strip() for r in args.resolutions.split(",") if r.strip()}

    def refs_on_disk(job: str) -> list[str]:
        d = root / "images" / job
        if not d.is_dir():
            return []
        pngs = list(d.glob("*.png")) + list(d.glob("*.jpg"))
        def key(p: Path):
            try:
                return (0, int(p.stem))
            except ValueError:
                return (1, p.stem)
        pngs = sorted(pngs, key=key)[: args.max_images]
        return [f"images/{job}/{p.name}" for p in pngs]

    c = sqlite3.connect(str(db))
    total = kept = 0
    drop = {"aspect": 0, "res": 0, "audio": 0, "novideo": 0, "noref": 0}
    with out.open("w", encoding="utf-8") as f:
        for job_id, data in c.execute("select job_id, data from manifest"):
            total += 1
            d = json.loads(data)
            rm = d.get("raw_metadata_json") or {}
            a, res, aud = rm.get("aspect_ratio"), rm.get("resolution"), bool(rm.get("generate_audio"))
            if a != args.aspect:
                drop["aspect"] += 1; continue
            if res not in keep_res:
                drop["res"] += 1; continue
            if args.require_audio and not aud:
                drop["audio"] += 1; continue
            if not (root / "videos" / f"{job_id}.mp4").is_file():
                drop["novideo"] += 1; continue
            refs = refs_on_disk(job_id)
            if not refs:
                drop["noref"] += 1; continue
            rec = {
                "job_id": job_id,
                "prompt": d.get("concatenated_prompt") or "",
                "aspect_ratio": a,
                "resolution": res,
                "total_duration_s": rm.get("duration"),
                "video": f"videos/{job_id}.mp4",
                "ref_images": refs,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"scanned {total:,} manifest rows -> kept {kept:,}", file=sys.stderr)
    print(f"dropped: {drop}", file=sys.stderr)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
