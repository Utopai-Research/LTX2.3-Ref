#!/usr/bin/env python3
"""
Preprocess the per-clip REFERENCE IMAGES (Seedance `ref_images`, 1-9 stills per clip) into
VAE latents for reference-image conditioning training (exp2).

Each reference image is assigned to its nearest-ASPECT bucket (16:9 -> 1376x768, 9:16 -> 768x1376,
other -> 768x768; see REF_BUCKETS), resized to that bucket, VAE-encoded, and patchified. Because
buckets differ in size, the K images per clip are encoded individually and their tokens concatenated.
One .pt per clip, keyed by the clip's VIDEO relative path so it pairs with latents/ conditions/ etc:

    <output-dir>/videos/<job_id>.pt  ->  {
        "ref_tokens":  Tensor[sum_k(h'_k*w'_k), C],  # all K images, patchified + concatenated
        "image_grids": Tensor[K, 2] (long),          # per-image latent (h', w') -> token count + positions
        "num_images":  K,
    }

Bucketing by nearest aspect + cover-crop means matched-aspect refs (~98% are 16:9/9:16) lose ~nothing,
unlike the old fixed-512x512 square crop that chopped ~44% off the tall 1536x2752 refs. --resize-mode
pad letterboxes into the bucket instead of cropping.

Multi-GPU: wrap with `accelerate launch --num_processes N` (each process handles an interleaved
shard). Resumable: existing .pt are skipped unless --overwrite. --dry-run enumerates only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import typer
from accelerate import PartialState
from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder
from ltx_trainer.utils import open_image_as_srgb

VAE_SPATIAL_FACTOR = 32

# Aspect-bucket reference encoding (v2): refs are ~98% 16:9 / 9:16, so assign each image to the
# nearest-aspect bucket and resize-cover-crop TO THAT BUCKET -> near-zero crop for matched-aspect
# images (vs the old fixed-512x512 square crop that chopped ~44% off the tall 1536x2752 refs).
# Each bucket is (W, H), both /32. Token count per image = (W/32)*(H/32) (varies by bucket).
REF_BUCKETS: dict[str, tuple[int, int]] = {
    "landscape": (1376, 768),  # 16:9  -> 43x24 = 1032 tokens
    "portrait": (768, 1376),   # 9:16  -> 24x43 = 1032 tokens
    "square": (768, 768),      # other -> 24x24 =  576 tokens
}


def _bucket_for(w: int, h: int) -> tuple[int, int]:
    """Pick the nearest-aspect bucket (W,H) for a w x h image."""
    ar = w / h
    if ar >= 1.3:
        return REF_BUCKETS["landscape"]
    if ar <= 1 / 1.3:
        return REF_BUCKETS["portrait"]
    return REF_BUCKETS["square"]

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _resize_crop(t: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """Resize maintaining AR so the image covers (th,tw), then center-crop. t: [C,H,W] in [0,1]."""
    _, h, w = t.shape
    scale = max(th / h, tw / w)
    nh, nw = max(round(h * scale), th), max(round(w * scale), tw)
    t = resize(t, [nh, nw], interpolation=InterpolationMode.BICUBIC, antialias=True)
    top, left = (nh - th) // 2, (nw - tw) // 2
    return t[:, top : top + th, left : left + tw]


def _resize_pad(t: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """Resize maintaining AR to fit inside (th,tw), then zero-pad (letterbox) to (th,tw)."""
    _, h, w = t.shape
    scale = min(th / h, tw / w)
    nh, nw = max(round(h * scale), 1), max(round(w * scale), 1)
    t = resize(t, [nh, nw], interpolation=InterpolationMode.BICUBIC, antialias=True)
    out = torch.zeros(t.shape[0], th, tw, dtype=t.dtype)
    top, left = (th - nh) // 2, (tw - nw) // 2
    out[:, top : top + nh, left : left + nw] = t
    return out


def _load_image(path: str, th: int, tw: int, mode: str) -> torch.Tensor:
    t = to_tensor(open_image_as_srgb(path)).clamp_(0, 1)  # [C,H,W] in [0,1]
    t = _resize_crop(t, th, tw) if mode == "crop" else _resize_pad(t, th, tw)
    return t * 2.0 - 1.0  # -> [-1, 1] (matches video preprocessing Normalize(0.5,0.5))


def _load_image_bucketed(path: str, mode: str) -> torch.Tensor:
    """Load image, pick its nearest-aspect bucket, resize to that bucket. Returns [C,th,tw] in [-1,1].

    For matched-aspect images (the ~98% that are 16:9/9:16) cover-crop removes ~nothing; only the
    rare odd-aspect images going to the 768x768 square bucket get a real crop (or letterbox if pad).
    """
    t = to_tensor(open_image_as_srgb(path)).clamp_(0, 1)  # [C,H,W] in [0,1]
    _, h, w = t.shape
    tw, th = _bucket_for(w, h)
    t = _resize_crop(t, th, tw) if mode == "crop" else _resize_pad(t, th, tw)
    return t * 2.0 - 1.0


def _atomic_save(data: object, out: Path) -> None:
    tmp = out.with_suffix(f"{out.suffix}.tmp.{os.getpid()}")
    torch.save(data, tmp)
    tmp.replace(out)


@app.command()
def main(  # noqa: PLR0913
    dataset_jsonl: str = typer.Argument(..., help="dataset.jsonl with `ref_images` + `video` fields"),
    model_path: str = typer.Option(..., help="LTX-2 checkpoint (for the video VAE encoder)"),
    output_dir: str = typer.Option(..., help="output dir (e.g. <root>/.precomputed_ti2v_16x9/reference_image_latents)"),
    manifest: str = typer.Option(None, help="optional jsonl of clips to restrict to (e.g. train_ti2v.jsonl, by job_id)"),
    resize_mode: str = typer.Option("crop", help="how to fit each image into its aspect bucket: crop "
                                    "(cover+center-crop, ~0 loss for matched-aspect refs) or pad (letterbox)"),
    ref_images_field: str = typer.Option("ref_images"),
    video_field: str = typer.Option("video"),
    max_images: int = typer.Option(9, help="cap reference images per clip"),
    device: str = typer.Option("cuda"),
    overwrite: bool = typer.Option(False),
    dry_run: bool = typer.Option(False, help="enumerate + report counts only; no model load / encode"),
) -> None:
    if resize_mode not in ("crop", "pad"):
        raise typer.BadParameter("resize-mode must be 'crop' or 'pad'")

    root = Path(dataset_jsonl).parent
    keep_ids: set[str] | None = None
    if manifest:
        keep_ids = {json.loads(line)["job_id"] for line in open(manifest) if line.strip()}

    # Build work list: (video_relpath, [existing ref image paths])
    clips: list[tuple[str, list[str]]] = []
    n_imgs = 0
    for line in open(dataset_jsonl):
        if not line.strip():
            continue
        d = json.loads(line)
        if keep_ids is not None and d.get("job_id") not in keep_ids:
            continue
        vid = d.get(video_field)
        refs = [p for p in (d.get(ref_images_field) or [])[:max_images] if (root / p).is_file()]
        if vid and refs:
            clips.append((vid, refs))
            n_imgs += len(refs)

    out_root = Path(output_dir)
    state = PartialState()
    shard = clips[state.process_index :: state.num_processes]
    todo = [c for c in shard if overwrite or not (out_root / Path(c[0]).with_suffix(".pt")).is_file()]
    if state.is_main_process:
        logger.info(f"{len(clips)} clips, {n_imgs} ref images total | aspect buckets {REF_BUCKETS} "
                    f"({resize_mode}) | output {out_root}")
    logger.info(f"Rank {state.process_index}/{state.num_processes}: {len(todo)} of {len(shard)} clips to do")

    if dry_run:
        return
    if not todo:
        return

    vae = load_video_vae_encoder(model_path, device=torch.device(device), dtype=torch.bfloat16)
    vae_dtype = next(vae.parameters()).dtype
    patchifier = VideoLatentPatchifier(patch_size=1)
    done = 0
    for vid, refs in todo:
        try:
            # Per-image: each ref goes to its own aspect bucket (different shapes), so encode + patchify
            # individually, then concatenate into one token sequence. image_grids records each image's
            # latent (h,w) so the trainer/eval can rebuild per-image slot ids + spatial positions.
            tokens: list[torch.Tensor] = []
            grids: list[tuple[int, int]] = []
            for p in refs:
                img = _load_image_bucketed(str(root / p), resize_mode)  # [C,th,tw] in [-1,1]
                video = img.unsqueeze(0).unsqueeze(2).to(device=device, dtype=vae_dtype)  # [1,C,1,th,tw]
                with torch.inference_mode():
                    lat = vae(video)  # [1, C, 1, h', w']
                tokens.append(patchifier.patchify(lat)[0].cpu())  # [h'*w', C] (bf16)
                grids.append((lat.shape[-2], lat.shape[-1]))       # (h', w')
            out = out_root / Path(vid).with_suffix(".pt")
            out.parent.mkdir(parents=True, exist_ok=True)
            _atomic_save(
                {
                    "ref_tokens": torch.cat(tokens, dim=0).contiguous(),  # [sum_k(h'_k*w'_k), C]
                    "image_grids": torch.tensor(grids, dtype=torch.long),  # [K, 2] = (h', w') per image
                    "num_images": len(refs),
                },
                out,
            )
            done += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed {vid}: {e}")
    logger.info(f"Rank {state.process_index}: encoded {done} clips -> {out_root}")


if __name__ == "__main__":
    app()
