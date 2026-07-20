#!/usr/bin/env python3
"""
Reference-image (exp2) evaluation harness for the Seedance experiments.

For each test clip, generates a video (+audio) conditioned on the clip's REFERENCE IMAGES
(images/<job_id>/{0,1,...}.png) + prompt — NO first frame. Each reference image is VAE-encoded
and concatenated in-context (see ValidationSampler._generate_with_reference_images).

GRID MODE (exp2 b/c/d/e matrix): --runs "name=<run_dir>@<step>,..." evaluates any number of
full-FT runs; each run's identity scheme (learned/sinusoidal/none embedding, shared vs per-image
RoPE time) is AUTO-CONFIGURED from <run_dir>/training_config.yaml so exp2c/d checkpoints are
rebuilt with the exact training-time scheme. The legacy --lora-steps/--full-steps flags still work.

REFERENCE CFG: --ref-guidance-scales "1.0,2.0,..." runs each (config, clip) at several scales.
scale 1.0 = off (no extra pass, output named <config>.mp4 — resume-compatible with previous
evals); scale != 1.0 adds a ref-less negative pass per step (output <config>_sref<scale>.mp4).

EFFICIENCY:
- (config, clip, scale) jobs are ordered config-major (each worker loads at most a couple of the
  36 GB checkpoints) and sharded CONTIGUOUSLY BY ESTIMATED COST (clip duration x 1.33 if the
  ref-CFG pass is on), so any number of workers = nodes x GPUs finishes together.
- Prompt embeddings are cached on disk once (<out>/.prompt_cache): after the first pass NO worker
  loads the 12B Gemma at all; transformer-only loads.
- Resume: an existing non-empty output mp4 is skipped unless --overwrite — nodes can die/relaunch.

Sharding: --worker W --num-workers N where W = node_idx*gpus + gpu (see run_eval_refs_grid.sh).
Match-GT length: each generation uses the GT clip's own frame count (snapped to %8==1).
Outputs: <out-dir>/<job_id>/<config>[_sref<scale>].mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import traceback
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torchvision import transforms

from ltx_trainer.model_loader import load_embeddings_processor, load_model, load_text_encoder
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import get_video_frame_count, save_video

VAE_TEMPORAL_FACTOR = 8
REF_EMB_KEY = "diffusion_model.reference_slot_embedding.weight"
REF_CFG_COST = 1.33  # ref-CFG adds a 4th forward pass per step (~+33% generation time)


class _TupleSafeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates the python/tuple tags pydantic's yaml.dump leaves in training_config.yaml."""


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: list(loader.construct_sequence(node))
)


def load_image(image_path: str) -> torch.Tensor:
    return transforms.ToTensor()(open_image_as_srgb(image_path))


def load_reference_images(images_root: str, job_id: str, max_refs: int) -> list[torch.Tensor]:
    """Load a clip's reference images (images/<job_id>/{0,1,...}.png), ordered so 0.png==[Image1]."""
    d = Path(images_root) / job_id
    if not d.is_dir():
        return []
    pngs = sorted(d.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1_000_000)
    return [load_image(str(p)) for p in pngs[:max_refs]]


def _extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    pat = re.compile(r"(.+)\.lora_[AB]\.")
    return sorted({m.group(1) for k in state_dict if (m := pat.match(k))})


def apply_lora(transformer: torch.nn.Module, lora_path: str | Path) -> torch.nn.Module:
    """Apply LoRA adapters from a checkpoint (ignores the non-LoRA reference embedding key)."""
    state_dict = load_file(str(lora_path))
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
    state_dict.pop("reference_slot_embedding.weight", None)  # base param, loaded separately
    target_modules = _extract_lora_target_modules(state_dict)
    if not target_modules:
        raise ValueError(f"No LoRA target modules found in {lora_path}")
    rank = next(v.shape[0] for k, v in state_dict.items() if "lora_A" in k and v.ndim == 2)
    cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules, lora_dropout=0.0, init_lora_weights=True)
    transformer = get_peft_model(transformer, cfg)
    set_peft_model_state_dict(transformer.get_base_model(), state_dict)
    return transformer


def load_reference_embedding_from_lora(transformer: torch.nn.Module, lora_path: str) -> bool:
    """Copy the trained reference identity embedding from a LoRA safetensors onto the base module."""
    sd = load_file(str(lora_path))
    if REF_EMB_KEY not in sd:
        return False
    base = transformer.get_base_model()
    with torch.no_grad():
        base.reference_slot_embedding.weight.copy_(sd[REF_EMB_KEY].to(base.reference_slot_embedding.weight))
    return True


def snap_frames(n: int) -> int:
    f = n - ((n - 1) % VAE_TEMPORAL_FACTOR)
    return max(f, 1)


def _parse_steps(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def _parse_scales(s: str) -> list[float]:
    scales = [float(x) for x in s.split(",") if x.strip()] if s else [1.0]
    return scales or [1.0]


def load_run_strategy(run_dir: Path, fallback: dict) -> dict:
    """Identity-scheme settings from a run's training_config.yaml (falls back to CLI defaults).

    This is what makes exp2c/d evaluable: eval MUST rebuild the training-time per-image RoPE time /
    no-embedding / sinusoidal scheme, or the outputs are silently wrong.
    """
    cfg_path = run_dir / "training_config.yaml"
    if not cfg_path.is_file():
        print(f"[warn] {cfg_path} missing -> using fallback strategy {fallback}", flush=True)
        return dict(fallback)
    ts = yaml.load(cfg_path.read_text(), Loader=_TupleSafeLoader)["training_strategy"]
    return {
        "embedding": ("none" if not ts.get("use_identity_embedding", True)
                      else ts.get("reference_embedding_type", "learned")),
        "time_per_image": ts.get("reference_time_per_image", False),
        "time_base": ts.get("reference_time_base", 20.0),
        "time_step": ts.get("reference_time_step", 1.0),
        "time_constant": ts.get("reference_time_constant", -1.0),
        "slots": ts.get("max_reference_slots", 9),
    }


def build_configs(args: argparse.Namespace) -> list[dict]:
    fallback = {
        "embedding": "learned", "time_per_image": False, "time_base": 20.0, "time_step": 1.0,
        "time_constant": args.ref_time_constant, "slots": args.ref_slots,
    }
    configs: list[dict] = []
    if args.include_base:
        configs.append({"name": "base", "lora": None, "full_weights": None,
                        "strategy": dict(fallback, embedding="none")})
    for s in _parse_steps(args.lora_steps):
        p = Path(args.lora_dir) / f"lora_weights_step_{s:05d}.safetensors"
        strategy = load_run_strategy(Path(args.lora_dir).parent, fallback)
        configs.append({"name": f"lora_{s}", "lora": str(p), "full_weights": None, "strategy": strategy})
    for s in _parse_steps(args.full_steps):
        p = Path(args.full_dir) / f"model_weights_step_{s:05d}.safetensors"
        strategy = load_run_strategy(Path(args.full_dir).parent, fallback)
        configs.append({"name": f"full_{s}", "lora": None, "full_weights": str(p), "strategy": strategy})
    for entry in [e.strip() for e in (args.runs or "").split(",") if e.strip()]:
        name, _, rest = entry.partition("=")
        run_dir, _, step = rest.partition("@")
        # tolerate line-wrapped/pasted RUNS strings: strip ALL whitespace inside each component
        name = "".join(name.split())
        run_dir = Path("".join(run_dir.split()))
        step = int("".join(step.split()) or "3000")
        ckpt = run_dir / "checkpoints" / f"model_weights_step_{step:05d}.safetensors"
        if not ckpt.is_file():
            raise FileNotFoundError(f"--runs entry {name!r}: {ckpt} not found")
        configs.append({"name": name, "lora": None, "full_weights": str(ckpt),
                        "strategy": load_run_strategy(run_dir, fallback)})
    names = [c["name"] for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate config names: {names}")
    return configs


def weighted_contiguous_shard(jobs: list, weights: list[float], worker: int, num_workers: int) -> list:
    """Split jobs into num_workers CONTIGUOUS segments of ~equal total weight; return segment `worker`.

    Contiguity preserves the config-major ordering (few model loads per worker); weighting equalizes
    wall-time despite 4-15s clip-length variance. Deterministic: every worker computes the same split.
    """
    if num_workers <= 1:
        return list(jobs)
    total = float(sum(weights))
    if total <= 0:
        chunk = math.ceil(len(jobs) / num_workers)
        return jobs[worker * chunk: (worker + 1) * chunk]
    shard, acc = [], 0.0
    for job, w in zip(jobs, weights):
        idx = min(int((acc + w / 2) / total * num_workers), num_workers - 1)
        if idx == worker:
            shard.append(job)
        acc += w
    return shard


def output_path(out_root: Path, job_id: str, config_name: str, scale: float) -> Path:
    suffix = "" if scale == 1.0 else f"_sref{scale:g}"
    return out_root / job_id / f"{config_name}{suffix}.mp4"


def segment_start_index(weights: list[float], worker: int, num_workers: int) -> int:
    """Index where this worker's contiguous weighted segment would begin (work-stealing locality anchor)."""
    if num_workers <= 1 or not weights:
        return 0
    total = float(sum(weights))
    if total <= 0:
        return min(len(weights), worker * math.ceil(len(weights) / num_workers))
    acc = 0.0
    for i, w in enumerate(weights):
        if min(int((acc + w / 2) / total * num_workers), num_workers - 1) >= worker:
            return i
        acc += w
    return 0


def try_claim(claim_path: Path, ttl_s: float) -> bool:
    """Atomically claim a job (O_CREAT|O_EXCL — verified atomic on the shared mount).

    Exactly one worker fleet-wide wins each claim, so workers can all scan the full job list
    and stay busy until the GLOBAL grid is done. A claim left by a dead/killed worker (older
    than ttl_s with no output) is stolen so crashes don't strand cells forever; a claim left
    by a [FAIL]ed job throttles retries to one attempt per ttl window.
    """
    for _ in range(2):  # second pass only after stealing a stale claim
        try:
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.uname().nodename}:{os.getpid()}:{int(time.time())}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - claim_path.stat().st_mtime
            except FileNotFoundError:
                continue  # released between open and stat -> retry the create
            if age < ttl_s:
                return False
            claim_path.unlink(missing_ok=True)  # stale claim from a dead worker: steal it
    return False


def _negative_cache_key(negative_prompt: str) -> str:
    return "negative_" + hashlib.sha1(negative_prompt.encode()).hexdigest()[:12]


def ensure_prompt_cache(
    cache_dir: Path, prompts: dict[str, str], args: argparse.Namespace, device: str, worker: int
) -> None:
    """Encode any uncached prompts to <cache_dir>/<key>.pt ({"video","audio"} bf16 cpu tensors).

    Loads Gemma ONLY if something is missing, then frees it — generation never needs a text encoder.
    Workers start at a strided offset and re-check existence before each encode, so concurrent
    first-run workers mostly encode disjoint subsets; atomic os.replace makes races harmless.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(k for k in prompts if not (cache_dir / f"{k}.pt").exists())
    if not keys:
        return
    offset = (worker * len(keys)) // max(args.num_workers, 1)
    keys = keys[offset:] + keys[:offset]
    print(f"[cache] up to {len(keys)} prompts to encode -> {cache_dir}", flush=True)
    text_encoder = load_text_encoder(args.text_encoder_path, device="cpu", dtype=torch.bfloat16)
    embeddings_processor = load_embeddings_processor(args.base_checkpoint, device="cpu", dtype=torch.bfloat16)
    text_encoder.to(device)
    embeddings_processor.to(device)
    encoded = 0
    with torch.inference_mode():
        for k in keys:
            final = cache_dir / f"{k}.pt"
            if final.exists():  # another worker got there first
                continue
            hs, mask = text_encoder.encode(prompts[k])
            out = embeddings_processor.process_hidden_states(hs, mask)
            payload = {
                "video": out.video_encoding.to("cpu", torch.bfloat16),
                "audio": out.audio_encoding.to("cpu", torch.bfloat16),
            }
            tmp = cache_dir / f".{k}.pt.tmp-{os.uname().nodename}-{os.getpid()}"  # PIDs collide ACROSS nodes
            torch.save(payload, tmp)
            os.replace(tmp, final)
            encoded += 1
    print(f"[cache] encoded {encoded} prompts", flush=True)
    del text_encoder, embeddings_processor
    torch.cuda.empty_cache()


def main() -> None:  # noqa: PLR0915, PLR0912
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-jsonl", default="/home/zheng/seedance_video_v0/test_set/test.jsonl")
    p.add_argument("--out-dir", default="/home/zheng/seedance_video_v0/eval_refs")
    p.add_argument("--images-root", default="/home/zheng/seedance_video_v0/images")
    p.add_argument("--base-checkpoint", default="/home/zheng/videogen/checkpoints/LTX-2.3/ltx-2.3-22b-dev.safetensors")
    p.add_argument("--text-encoder-path", default="/home/zheng/videogen/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized")
    # model grid: --runs (auto-configured full-FT runs) and/or the legacy lora/full flags
    p.add_argument("--runs", default="", help='comma list "name=<run_dir>@<step>"; identity scheme read '
                                              "from <run_dir>/training_config.yaml")
    p.add_argument("--lora-dir", default="/home/zheng/seedance_video_v0/runs/ltx2.3_reference_lora/checkpoints")
    p.add_argument("--full-dir", default="/home/zheng/seedance_video_v0/runs/ltx2.3_reference_full/checkpoints")
    p.add_argument("--lora-steps", default="", help="comma list, e.g. 1000,2000,3000")
    p.add_argument("--full-steps", default="", help="comma list, e.g. 1000,2000,3000")
    p.add_argument("--include-base", action="store_true", help="also run the base model (no trained ref embedding)")
    # reference-image conditioning fallbacks (used only when a run has no training_config.yaml)
    p.add_argument("--ref-slots", type=int, default=9, help="max_reference_slots (identity-embedding size)")
    p.add_argument("--max-refs", type=int, default=9, help="cap reference images per clip")
    p.add_argument("--ref-resolution", type=int, default=512, help="square px each ref image is encoded at")
    p.add_argument("--ref-bucketed", action="store_true",
                   help="v2 models: encode each ref into its nearest-aspect bucket (16:9->1376x768, "
                        "9:16->768x1376, other->768x768) instead of the square --ref-resolution crop. "
                        "MUST match how the model was trained (v2 precompute is bucketed).")
    p.add_argument("--ref-time-constant", type=float, default=-1.0)
    # reference CFG
    p.add_argument("--ref-guidance-scales", default="1.0",
                   help="comma list; 1.0 = off (plain <config>.mp4), else <config>_sref<scale>.mp4. "
                        "0 = refs fully ignored (T2V-retention diagnostic).")
    # generation settings
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--frame-rate", type=float, default=24.0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = match GT length; else cap")
    p.add_argument("--inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--stg-scale", type=float, default=1.0)
    p.add_argument("--stg-blocks", type=int, nargs="*", default=[29])
    p.add_argument("--stg-mode", default="stg_av", choices=["stg_av", "stg_v"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-audio", action="store_true")
    # sharding / control
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="limit clips (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--prompt-cache-dir", default="", help="default: <out-dir>.prompt_cache (sibling, NOT "
                   "inside out-dir, so it's never pulled into the synced gallery)")
    # node staging (used by run_eval_refs_grid.sh before spawning workers)
    p.add_argument("--cache-only", action="store_true",
                   help="encode ALL manifest prompts into the cache (one Gemma load), then exit "
                        "(unless --plan-node is also given)")
    p.add_argument("--plan-node", type=int, default=-1,
                   help="print the checkpoint paths needed by this node's workers (for page-cache "
                        "warming), then exit")
    p.add_argument("--plan-gpus", type=int, default=8, help="workers per node for --plan-node")
    p.add_argument("--claim-ttl", type=float, default=2700.0,
                   help="seconds before another worker may steal a job's claim (crash recovery)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    generate_audio = not args.skip_audio
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    clips = [json.loads(line) for line in open(args.test_jsonl) if line.strip()]
    if args.limit:
        clips = clips[: args.limit]
    configs = build_configs(args)
    if not configs:
        raise SystemExit("No configs selected (use --runs and/or --include-base/--lora-steps/--full-steps)")
    scales = _parse_scales(args.ref_guidance_scales)

    # (config, clip, scale) jobs: config-major (few model loads/worker), cost-weighted contiguous shards.
    jobs = [(ci, clip, s) for ci in range(len(configs)) for clip in clips for s in scales]
    weights = [float(clip.get("duration_s") or 5.0) * (REF_CFG_COST if s != 1.0 else 1.0) for _, clip, s in jobs]

    out_root = Path(args.out_dir)
    # Keep the prompt cache OUTSIDE out_root (which doubles as a synced frontend gallery) -> sibling dir.
    cache_dir = Path(args.prompt_cache_dir) if args.prompt_cache_dir else out_root.parent / f"{out_root.name}.prompt_cache"
    neg_key = _negative_cache_key(negative_prompt)

    def is_done(ci: int, clip: dict, scale: float) -> bool:
        op = output_path(out_root, clip["job_id"], configs[ci]["name"], scale)
        return not args.overwrite and op.exists() and op.stat().st_size > 0

    # --- node staging modes (run once per node by the launcher, before the workers spawn) ---
    if args.cache_only:
        # Encode EVERY manifest prompt up front: at most ONE Gemma load per node (instead of one
        # per worker on a cold cache), and zero on any later launch.
        prompts = {c["job_id"]: c["caption"] for c in clips}
        prompts[neg_key] = negative_prompt
        ensure_prompt_cache(cache_dir, prompts, args, args.device, worker=0)
        print("[cache-only] prompt cache complete", flush=True)
        if args.plan_node < 0:
            return
    if args.plan_node >= 0:
        # Print the unique checkpoint files this node's PENDING jobs will load, so the launcher can
        # warm the kernel page cache with one sequential read each (measured: re-reads then run at
        # ~5 GB/s instead of 8 workers contending on ~0.4 GB/s network streams).
        needed: dict[str, None] = {}
        for w in range(args.plan_node * args.plan_gpus, (args.plan_node + 1) * args.plan_gpus):
            for ci, clip, s in weighted_contiguous_shard(jobs, weights, w, args.num_workers):
                if is_done(ci, clip, s):
                    continue
                for key in ("full_weights", "lora"):
                    if configs[ci].get(key):
                        needed[configs[ci][key]] = None
        if needed:  # base checkpoint is needed by every config that has pending work
            print(args.base_checkpoint, flush=True)
        for pth in needed:
            print(pth, flush=True)
        return

    # Work-stealing execution: every worker scans the FULL job list, starting at its own static
    # segment (locality: the bulk of its work is the same configs the contiguous partition would
    # give it -> ~1 checkpoint load), claiming each pending job atomically. Workers that exhaust
    # their region roll into whatever is still unclaimed — every GPU stays busy until the GLOBAL
    # grid is done, regardless of already-done cells, node count, or when nodes join.
    start = segment_start_index(weights, args.worker, args.num_workers)
    ordered = jobs[start:] + jobs[:start]
    pending = [j for j in ordered if not is_done(*j)]
    print(f"[worker {args.worker}/{args.num_workers}] start_offset={start}/{len(jobs)} "
          f"global_pending={len(pending)} scales={scales}", flush=True)

    # Prompt-embedding cache: after this block no worker ever loads Gemma again.
    needed_prompts = {clip["job_id"]: clip["caption"] for _, clip, _ in pending}
    needed_prompts[neg_key] = negative_prompt
    if pending:
        ensure_prompt_cache(cache_dir, needed_prompts, args, args.device, args.worker)

    emb_cache: dict[str, dict] = {}

    def _load_cache_entry(key: str) -> dict:
        if key not in emb_cache:
            f = cache_dir / f"{key}.pt"
            try:
                emb_cache[key] = torch.load(f, map_location="cpu", weights_only=True)
            except Exception:
                f.unlink(missing_ok=True)  # corrupt entry: drop it so a relaunch re-encodes it
                raise
        return emb_cache[key]

    def cached_embeddings_for(job_id: str) -> CachedPromptEmbeddings:
        pos, neg = _load_cache_entry(job_id), _load_cache_entry(neg_key)
        return CachedPromptEmbeddings(
            video_context_positive=pos["video"], audio_context_positive=pos["audio"],
            video_context_negative=neg["video"], audio_context_negative=neg["audio"],
        )

    loaded_ci = None
    sampler = components = None

    for ci, clip, scale in pending:
        cfg = configs[ci]
        st = cfg["strategy"]
        jid = clip["job_id"]
        out_path = output_path(out_root, jid, cfg["name"], scale)
        if is_done(ci, clip, scale):  # done by another worker since our startup scan
            continue

        ref_imgs = load_reference_images(args.images_root, jid, min(args.max_refs, st["slots"]))
        if not ref_imgs:
            print(f"[skip] {jid}/{cfg['name']} (no reference images under {args.images_root}/{jid})", flush=True)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path = out_path.parent / (out_path.name + ".claim")
        if not try_claim(claim_path, args.claim_ttl):
            continue  # another worker owns this job
        if is_done(ci, clip, scale):  # claim raced with a completed steal: nothing to do
            claim_path.unlink(missing_ok=True)
            continue

        if ci != loaded_ci:
            del sampler, components
            torch.cuda.empty_cache()
            is_ft = bool(cfg["lora"] or cfg.get("full_weights"))
            print(f"[load] config={cfg['name']} strategy={st}", flush=True)
            components = load_model(
                checkpoint_path=args.base_checkpoint,
                device="cpu",
                dtype=torch.bfloat16,
                with_video_vae_encoder=True,   # needed to encode reference images
                with_video_vae_decoder=True,
                with_audio_vae_decoder=generate_audio,
                with_vocoder=generate_audio,
                with_text_encoder=False,       # prompts come from the on-disk embedding cache
            )
            transformer = components.transformer
            # Create the identity-embedding module on fine-tuned configs (BEFORE LoRA wrap / state load).
            # "none" (exp2c) = no module; "sinusoidal" (exp2d) = non-persistent buffer regenerated here.
            if is_ft and st["embedding"] != "none":
                transformer.enable_reference_embedding(st["slots"], st["embedding"])
            if cfg["lora"]:
                transformer = apply_lora(transformer, cfg["lora"])
                if st["embedding"] == "learned":
                    ok = load_reference_embedding_from_lora(transformer, cfg["lora"])
                    print(f"[lora] reference embedding loaded: {ok}", flush=True)
            elif cfg.get("full_weights"):
                size_gb = Path(cfg["full_weights"]).stat().st_size / 1e9
                print(f"[load] reading {size_gb:.0f} GB checkpoint (silent step; fast if page-cache "
                      f"warmed, minutes over cold network)...", flush=True)
                sd = torch.load(cfg["full_weights"], map_location="cpu", weights_only=True)
                missing, unexpected = transformer.load_state_dict(sd, strict=False)
                print(f"[full] loaded {Path(cfg['full_weights']).name}: missing={len(missing)} "
                      f"unexpected={len(unexpected)} (both ~0)", flush=True)
                del sd
            sampler = ValidationSampler(
                transformer=transformer,
                vae_decoder=components.video_vae_decoder,
                vae_encoder=components.video_vae_encoder,
                text_encoder=None,
                audio_decoder=components.audio_vae_decoder if generate_audio else None,
                vocoder=components.vocoder if generate_audio else None,
            )
            loaded_ci = ci

        try:
            n = get_video_frame_count(clip["video"])
        except Exception:
            n = int(clip.get("duration_s", 5)) * int(round(args.frame_rate)) + 1
        num_frames = snap_frames(n if args.max_frames == 0 else min(n, args.max_frames))
        width = int(clip.get("width", args.width))
        height = int(clip.get("height", args.height))

        print(f"[gen ] {jid}/{out_path.stem}  {width}x{height}x{num_frames}  refs={len(ref_imgs)} "
              f"sref={scale:g}", flush=True)
        try:
            gen = GenerationConfig(
                prompt=clip["caption"],
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                reference_images=ref_imgs,
                reference_slot_count=st["slots"],
                reference_time_constant=st["time_constant"],
                reference_time_per_image=st["time_per_image"],
                reference_time_base=st["time_base"],
                reference_time_step=st["time_step"],
                reference_resolution=args.ref_resolution,
                reference_bucketed=args.ref_bucketed,
                ref_guidance_scale=scale,
                cached_embeddings=cached_embeddings_for(jid),
                generate_audio=generate_audio,
                stg_scale=args.stg_scale,
                stg_blocks=args.stg_blocks,
                stg_mode=args.stg_mode,
            )
            video, audio = sampler.generate(config=gen, device=args.device)
            sr = components.vocoder.output_sampling_rate if (audio is not None and components.vocoder) else None
            # save_video muxes into the target path incrementally: write to a tmp name and rename,
            # so a killed worker can never leave a truncated mp4 that resume would treat as done.
            # (hidden tmp keeps gallery globs clean; .mp4 suffix kept so av infers the container)
            tmp_path = out_path.parent / f".{out_path.stem}.{os.uname().nodename}-{os.getpid()}.tmp.mp4"
            save_video(video_tensor=video, output_path=tmp_path, fps=args.frame_rate,
                       audio=audio, audio_sample_rate=sr)
            os.replace(tmp_path, out_path)
            claim_path.unlink(missing_ok=True)
            print(f"[done] {out_path}", flush=True)
        except Exception:
            # Leave the claim in place: one attempt per claim-ttl window fleet-wide, so a
            # deterministic failure isn't retried by all 48 workers in a loop.
            print(f"[FAIL] {jid}/{out_path.stem}\n{traceback.format_exc()}", flush=True)

    print(f"[worker {args.worker}/{args.num_workers}] finished", flush=True)


if __name__ == "__main__":
    main()
