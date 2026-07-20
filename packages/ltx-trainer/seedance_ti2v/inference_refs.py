#!/usr/bin/env python3
"""
Standalone reference-to-video (R2V) inference for a trained exp2 checkpoint.

Generates ONE video from a text prompt + 0-9 reference images (no first frame), using
a fine-tuned LTX-2.3 checkpoint (full-FT or LoRA — auto-detected from the weights file).
With no --ref-images it runs pure text-to-video (the ref-dropout-trained models keep
a calibrated ref-less branch).

The reference-conditioning scheme (learned/sinusoidal/no identity embedding, shared vs
per-image RoPE time, slot count) MUST match training. It is auto-configured from the
run's training_config.yaml when one is found next to the checkpoint (or passed via
--training-config); otherwise the CLI fallbacks are used.

Requirements (see download_models.sh):
  - LTX-2.3 base checkpoint  (ltx-2.3-22b-dev.safetensors, ~44 GB)
  - Gemma-3 text encoder dir (~24 GB)
  - your trained checkpoint  (e.g. model_weights_step_03000.safetensors from Google Drive)

Example (v2 model, 1280x704):
  python inference_refs.py \
      --checkpoint /path/to/model_weights_step_03000.safetensors \
      --base-checkpoint /path/to/LTX-2.3/ltx-2.3-22b-dev.safetensors \
      --text-encoder-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
      --ref-images ref0.png ref1.png \
      --prompt "A woman in a red coat walks through a snowy street, camera tracking." \
      --output out.mp4

v1 (768x448, square 512 refs) checkpoints: add --ref-mode square --width 768 --height 448.
VRAM: a single >=80 GB GPU (H100/H200/A100-80G) is expected for the 22B model in bf16.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torchvision import transforms

from ltx_trainer.model_loader import load_embeddings_processor, load_model, load_text_encoder
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import save_video

VAE_TEMPORAL_FACTOR = 8
REF_EMB_KEY = "diffusion_model.reference_slot_embedding.weight"


class _TupleSafeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates the python/tuple tags pydantic's yaml.dump leaves in training_config.yaml."""


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: list(loader.construct_sequence(node))
)


def snap_frames(n: int) -> int:
    """Snap a frame count down to the nearest valid LTX length (frames % 8 == 1)."""
    return max(n - ((n - 1) % VAE_TEMPORAL_FACTOR), 1)


def load_state_dict_any(path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint saved either as real safetensors or via torch.save.

    The trainer's full-FT checkpoints are torch.save zip archives despite their
    .safetensors filename — detect by magic bytes, NOT by extension. (Feeding a zip
    to safetensors' mmap parser can SIGBUS on network filesystems, so no try/except.)
    """
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"PK":  # zip -> torch.save format
        return torch.load(path, map_location="cpu", weights_only=True)
    return load_file(str(path))


def is_lora_state_dict(sd: dict[str, torch.Tensor]) -> bool:
    return any(".lora_A." in k or ".lora_B." in k for k in sd)


def apply_lora(transformer: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Wrap the transformer with LoRA adapters reconstructed from the checkpoint and load them."""
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
    ref_emb = state_dict.pop("reference_slot_embedding.weight", None)  # base param, loaded after wrap
    pat = re.compile(r"(.+)\.lora_[AB]\.")
    target_modules = sorted({m.group(1) for k in state_dict if (m := pat.match(k))})
    if not target_modules:
        raise ValueError("No LoRA target modules found in checkpoint")
    rank = next(v.shape[0] for k, v in state_dict.items() if "lora_A" in k and v.ndim == 2)
    cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules, lora_dropout=0.0, init_lora_weights=True)
    transformer = get_peft_model(transformer, cfg)
    set_peft_model_state_dict(transformer.get_base_model(), state_dict)
    if ref_emb is not None:
        base = transformer.get_base_model()
        with torch.no_grad():
            base.reference_slot_embedding.weight.copy_(ref_emb.to(base.reference_slot_embedding.weight))
    return transformer


def resolve_strategy(args: argparse.Namespace) -> dict:
    """Reference-conditioning scheme from training_config.yaml (auto-discovered) or CLI fallbacks."""
    cfg_path = Path(args.training_config) if args.training_config else None
    if cfg_path is None:
        ckpt = Path(args.checkpoint)
        for cand in (ckpt.parent / "training_config.yaml", ckpt.parent.parent / "training_config.yaml"):
            if cand.is_file():
                cfg_path = cand
                break
    if cfg_path is None or not cfg_path.is_file():
        print(f"[warn] no training_config.yaml found -> using CLI fallbacks "
              f"(embedding={args.ref_embedding}, slots={args.ref_slots})", flush=True)
        return {
            "embedding": args.ref_embedding, "time_per_image": args.ref_time_per_image,
            "time_base": args.ref_time_base, "time_step": args.ref_time_step,
            "time_constant": args.ref_time_constant, "slots": args.ref_slots,
        }
    ts = yaml.load(cfg_path.read_text(), Loader=_TupleSafeLoader)["training_strategy"]
    strategy = {
        "embedding": ("none" if not ts.get("use_identity_embedding", True)
                      else ts.get("reference_embedding_type", "learned")),
        "time_per_image": ts.get("reference_time_per_image", False),
        "time_base": ts.get("reference_time_base", 20.0),
        "time_step": ts.get("reference_time_step", 1.0),
        "time_constant": ts.get("reference_time_constant", -1.0),
        "slots": ts.get("max_reference_slots", 9),
    }
    print(f"[cfg ] strategy from {cfg_path}: {strategy}", flush=True)
    return strategy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="trained weights: full-FT model_weights_*.safetensors or LoRA lora_weights_*.safetensors")
    p.add_argument("--base-checkpoint", required=True, help="LTX-2.3 base ltx-2.3-22b-dev.safetensors")
    p.add_argument("--text-encoder-path", required=True, help="Gemma-3 text encoder directory")
    p.add_argument("--training-config", default="",
                   help="run's training_config.yaml (default: auto-search next to --checkpoint)")
    # inputs / output
    p.add_argument("--ref-images", nargs="*", default=[],
                   help="0-9 reference image paths, order = [Image1..N]; omit for pure text-to-video")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="worst quality, inconsistent motion, blurry, jittery, distorted")
    p.add_argument("--output", default="output.mp4")
    # generation settings (defaults = v2 model: 1280x704, bucketed refs)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--num-frames", type=int, default=121, help="snapped down to %%8==1")
    p.add_argument("--frame-rate", type=float, default=24.0)
    p.add_argument("--inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--ref-guidance-scale", type=float, default=1.0,
                   help="reference CFG (needs a ref-dropout-trained model, e.g. exp2e); 1.0 = off")
    p.add_argument("--stg-scale", type=float, default=1.0)
    p.add_argument("--stg-blocks", type=int, nargs="*", default=[29])
    p.add_argument("--stg-mode", default="stg_av", choices=["stg_av", "stg_v"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-audio", action="store_true")
    # reference encoding — MUST match training
    p.add_argument("--ref-mode", default="bucketed", choices=["bucketed", "square"],
                   help="bucketed = v2 aspect buckets (16:9->1376x768, 9:16->768x1376, other->768x768); "
                        "square = v1 fixed square crop at --ref-resolution")
    p.add_argument("--ref-resolution", type=int, default=512, help="square px per ref image (--ref-mode square)")
    # scheme fallbacks (used only when no training_config.yaml is found)
    p.add_argument("--ref-embedding", default="learned", choices=["learned", "sinusoidal", "none"])
    p.add_argument("--ref-slots", type=int, default=9)
    p.add_argument("--ref-time-constant", type=float, default=-1.0)
    p.add_argument("--ref-time-per-image", action="store_true")
    p.add_argument("--ref-time-base", type=float, default=20.0)
    p.add_argument("--ref-time-step", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.width % 32 or args.height % 32:
        raise SystemExit("--width/--height must be divisible by 32")
    strategy = resolve_strategy(args)
    if len(args.ref_images) > strategy["slots"]:
        raise SystemExit(f"{len(args.ref_images)} reference images exceeds the model's {strategy['slots']} slots")
    generate_audio = not args.skip_audio

    ref_imgs = [transforms.ToTensor()(open_image_as_srgb(p_)) for p_ in args.ref_images]
    print(f"[load] base model from {args.base_checkpoint}", flush=True)
    components = load_model(
        checkpoint_path=args.base_checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=True,   # encodes the reference images
        with_video_vae_decoder=True,
        with_audio_vae_decoder=generate_audio,
        with_vocoder=generate_audio,
        with_text_encoder=False,       # Gemma loaded separately below
    )
    transformer = components.transformer
    if strategy["embedding"] != "none":
        transformer.enable_reference_embedding(strategy["slots"], strategy["embedding"])

    print(f"[load] trained weights from {args.checkpoint}", flush=True)
    sd = load_state_dict_any(Path(args.checkpoint))
    if is_lora_state_dict(sd):
        print("[load] LoRA checkpoint detected", flush=True)
        transformer = apply_lora(transformer, sd)
    else:
        missing, unexpected = transformer.load_state_dict(sd, strict=False)
        print(f"[load] full checkpoint: missing={len(missing)} unexpected={len(unexpected)} (both should be ~0)",
              flush=True)
    del sd

    print(f"[load] text encoder from {args.text_encoder_path}", flush=True)
    text_encoder = load_text_encoder(args.text_encoder_path, device="cpu", dtype=torch.bfloat16)
    embeddings_processor = load_embeddings_processor(args.base_checkpoint, device="cpu", dtype=torch.bfloat16)

    sampler = ValidationSampler(
        transformer=transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=text_encoder,
        audio_decoder=components.audio_vae_decoder if generate_audio else None,
        vocoder=components.vocoder if generate_audio else None,
        embeddings_processor=embeddings_processor,
    )

    num_frames = snap_frames(args.num_frames)
    print(f"[gen ] {args.width}x{args.height}x{num_frames} refs={len(ref_imgs)} seed={args.seed}", flush=True)
    gen = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        reference_images=ref_imgs or None,
        reference_slot_count=strategy["slots"],
        reference_time_constant=strategy["time_constant"],
        reference_time_per_image=strategy["time_per_image"],
        reference_time_base=strategy["time_base"],
        reference_time_step=strategy["time_step"],
        reference_resolution=args.ref_resolution,
        reference_bucketed=args.ref_mode == "bucketed",
        ref_guidance_scale=args.ref_guidance_scale,
        generate_audio=generate_audio,
        stg_scale=args.stg_scale,
        stg_blocks=args.stg_blocks,
        stg_mode=args.stg_mode,
    )
    video, audio = sampler.generate(config=gen, device=args.device)
    sr = components.vocoder.output_sampling_rate if (audio is not None and components.vocoder) else None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video_tensor=video, output_path=out_path, fps=args.frame_rate, audio=audio, audio_sample_rate=sr)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
