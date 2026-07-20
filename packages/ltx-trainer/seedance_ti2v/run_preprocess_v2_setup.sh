#!/usr/bin/env bash
# Create the v2 precompute root and symlink the resolution-INDEPENDENT sources from v1.
# Text conditions (Gemma embeds, 30G) and audio latents (382M) do not depend on video
# resolution, so v2 reuses them via symlink. Only latents/ (1280x704 video) and
# reference_image_latents/ (aspect-bucketed) are recomputed into the v2 root.
#
# Run this ONCE before training. Idempotent: re-running just re-points the symlinks.
set -euo pipefail

V1="${V1:-/home/zheng/seedance_video_v0/.precomputed_ti2v_16x9}"
V2="${V2:-/home/zheng/seedance_video_v0/.precomputed_ti2v_16x9_v2}"

mkdir -p "$V2"
# Resolution-independent -> symlink from v1.
ln -sfnT "$V1/conditions"    "$V2/conditions"
ln -sfnT "$V1/audio_latents" "$V2/audio_latents"

echo ">> v2 root: $V2"
echo "   conditions    -> $(readlink "$V2/conditions")"
echo "   audio_latents -> $(readlink "$V2/audio_latents")"
echo "   latents/                 <- run_preprocess_v2_video.sh (1280x704)"
echo "   reference_image_latents/ <- OUT_DIR=$V2/reference_image_latents run_preprocess_refs.sh"
