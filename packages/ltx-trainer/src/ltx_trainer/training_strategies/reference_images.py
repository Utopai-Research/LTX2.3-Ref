"""Reference-image conditioning training strategy (exp2).

Conditions video (+ optional audio) generation on N reference images (the Seedance `ref_images`,
1-9 per clip), concatenated in-context into the video token sequence (like `video_to_video`),
with a learned/sinusoidal per-image identity embedding (handled by LTXModel via
`Modality.reference_slot_ids`). Positions for reference tokens (option 3):
  - spatial (h,w): RoPE on each image's own grid,
  - time: a single fixed reserved constant (identity comes from the per-image embedding, not time).
Reference tokens are clean (timestep 0); loss is on the target video (excluding refs) + all audio.
Audio handling mirrors `text_to_video` (joint AV).

Reference dropout (CFG-style condition dropout):
  - `reference_dropout_p`: per-sample probability of dropping ALL reference images, so the model
    also learns p(video | text) and a "no refs" negative branch becomes valid at inference
    (reference CFG, analogous to the image branch of InstructPix2Pix-style two-condition CFG).
  - `reference_image_dropout_p`: independent per-image dropout (>=1 image always kept). Surviving
    images keep their ORIGINAL slot index / RoPE time, so the [ImageK] <-> slot-k binding never shifts.
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class ReferenceImagesConfig(TrainingStrategyConfigBase):
    """Configuration for reference-image conditioning."""

    name: Literal["reference_images"] = "reference_images"

    reference_latents_dir: str = Field(
        default="reference_image_latents",
        description="Directory (under preprocessed_data_root) with per-clip reference-image latents.",
    )
    max_reference_slots: int = Field(
        default=9, ge=1,
        description="Max reference images per clip; also the size of the model's identity embedding table.",
    )
    reference_time_constant: float = Field(
        default=-1.0,
        description="Fixed RoPE time coordinate assigned to all reference tokens (out of the video's range).",
    )
    reference_embedding_type: Literal["learned", "sinusoidal"] = Field(
        default="learned",
        description="Identity embedding type. Passed to LTXModel (num_reference_slots/reference_embedding_type).",
    )
    # --- PE ablations (exp2c / exp2d) ---
    use_identity_embedding: bool = Field(
        default=True,
        description="If False, NO per-image identity embedding is created (identity carried by RoPE time only -> exp2c).",
    )
    reference_time_per_image: bool = Field(
        default=False,
        description="If True, image-k gets a distinct RoPE time (base + step*k) instead of a shared constant (exp2c).",
    )
    reference_time_base: float = Field(
        default=20.0,
        description="RoPE time for image 0 when reference_time_per_image (20 = a reserved block after the video).",
    )
    reference_time_step: float = Field(
        default=1.0,
        description="RoPE time increment per image when reference_time_per_image (+1 -> 20,21,..; with base -1, -1 -> -1,-2,..).",
    )
    # --- reference dropout (CFG-style condition dropout) ---
    reference_dropout_p: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Per-sample probability of dropping ALL reference images (trains the ref-unconditional "
        "branch p(video | text), enabling reference CFG at inference).",
    )
    reference_image_dropout_p: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Independent per-image dropout probability (>=1 image always kept; survivors keep their "
        "ORIGINAL slot index / RoPE time so the [ImageK] <-> slot-k binding never shifts).",
    )
    first_frame_conditioning_p: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of ALSO giving the target's first frame as conditioning (default off).",
    )
    with_audio: bool = Field(default=True, description="Joint audio-video training (mirrors text_to_video).")
    audio_latents_dir: str = Field(default="audio_latents")


class ReferenceImagesStrategy(TrainingStrategy):
    """Reference-image conditioning (+ optional joint audio)."""

    config: ReferenceImagesConfig

    def __init__(self, config: ReferenceImagesConfig):
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    def get_data_sources(self) -> dict[str, str]:
        sources = {
            "latents": "latents",
            "conditions": "conditions",
            self.config.reference_latents_dir: "ref_image_latents",
        }
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "max_reference_slots": self.config.max_reference_slots,
            "reference_time_constant": self.config.reference_time_constant,
            "reference_embedding_type": self.config.reference_embedding_type,
            "use_identity_embedding": self.config.use_identity_embedding,
            "reference_time_per_image": self.config.reference_time_per_image,
            "reference_time_base": self.config.reference_time_base,
            "reference_time_step": self.config.reference_time_step,
            "reference_dropout_p": self.config.reference_dropout_p,
            "reference_image_dropout_p": self.config.reference_image_dropout_p,
        }

    def prepare_training_inputs(  # noqa: PLR0915
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        latents = batch["latents"]
        target_latents = latents["latents"]  # [B, C, F, H, W]
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()
        fps = latents.get("fps", None)
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        device = target_latents.device
        dtype = target_latents.dtype
        batch_size = target_latents.shape[0]

        # --- reference images: PRE-PATCHIFIED, per aspect bucket -> variable tokens per image ---
        # payload (process_reference_images.py): ref_tokens [B, total, C] (K images concatenated),
        # image_grids [B, K, 2] = each image's latent (h, w). Images may differ in token count.
        ref_all = batch["ref_image_latents"]["ref_tokens"]
        if ref_all.dim() != 3:  # safety
            raise ValueError(f"ref_tokens must be [B, total, C], got {tuple(ref_all.shape)}")
        ref_all = ref_all[0].to(dtype)  # [total, C]
        grids = batch["ref_image_latents"]["image_grids"][0]  # [K, 2] = (h, w) per image
        per_img_tokens = (grids[:, 0] * grids[:, 1]).tolist()  # token count per image
        offsets = [0]
        for n in per_img_tokens:
            offsets.append(offsets[-1] + n)

        # Reference dropout: drop ALL refs (trains the ref-unconditional branch -> reference CFG) or
        # per-image (>=1 kept). Survivors keep their ORIGINAL slot index so the [ImageK] <-> slot-k
        # binding (identity embedding / per-image RoPE time) never shifts.
        slot_indices = torch.arange(grids.shape[0], device=device)
        if self.config.reference_dropout_p > 0 and torch.rand(1, device=device).item() < self.config.reference_dropout_p:
            slot_indices = slot_indices[:0]
        elif self.config.reference_image_dropout_p > 0 and slot_indices.numel() > 1:
            keep = torch.rand(slot_indices.shape, device=device) >= self.config.reference_image_dropout_p
            if not keep.any():  # always keep >=1: dropping ALL refs is reference_dropout_p's job
                keep[torch.randint(keep.numel(), (1,), device=device)] = True
            slot_indices = slot_indices[keep]

        # Assemble kept reference tokens + per-token slot ids + per-token positions, image by image
        # (each image uses its OWN h,w spatial grid; time per the identity scheme).
        empty_pos = self._get_video_positions(1, 1, 1, batch_size, fps, device, dtype)[:, :, :0, :]
        ref_token_parts: list[Tensor] = [ref_all[:0]]  # empty seed -> valid cat if all refs dropped
        ref_slot_id_parts: list[Tensor] = []
        ref_pos_parts: list[Tensor] = [empty_pos]
        for k in slot_indices.tolist():
            h_k, w_k = int(grids[k, 0]), int(grids[k, 1])
            ref_token_parts.append(ref_all[offsets[k] : offsets[k + 1]])  # [tpi_k, C]
            ref_slot_id_parts.append(torch.full((per_img_tokens[k],), k, dtype=torch.long, device=device))
            p = self._get_video_positions(
                num_frames=1, height=h_k, width=w_k, batch_size=batch_size, fps=fps, device=device, dtype=dtype,
            ).clone()  # [B, 3, h_k*w_k, 2]
            if self.config.reference_time_per_image:
                p[:, 0, :, :] = self.config.reference_time_base + self.config.reference_time_step * k
            else:
                p[:, 0, :, :] = self.config.reference_time_constant
            ref_pos_parts.append(p)
        ref_tokens = torch.cat(ref_token_parts, dim=0).unsqueeze(0)  # [1, ref_seq_len, C]
        ref_seq_len = ref_tokens.shape[1]
        ref_slot_ids = (
            torch.cat(ref_slot_id_parts) if ref_slot_id_parts else torch.empty(0, dtype=torch.long, device=device)
        )
        ref_positions = torch.cat(ref_pos_parts, dim=2)  # [B, 3, ref_seq_len, 2]

        # --- patchify target video ---
        target_latents = self._video_patchifier.patchify(target_latents)  # [B, T, C]
        target_seq_len = target_latents.shape[1]

        # --- text embeddings (connectors already applied by the trainer) ---
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # --- conditioning mask: refs always clean; target optionally first-frame ---
        ref_cond_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=device)
        target_cond_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size, sequence_length=target_seq_len, height=height, width=width,
            device=device, first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )
        conditioning_mask = torch.cat([ref_cond_mask, target_cond_mask], dim=1)

        # --- noise the target ---
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise
        noisy_target = torch.where(target_cond_mask.unsqueeze(-1), target_latents, noisy_target)
        video_targets = noise - target_latents  # velocity target (target tokens only)

        combined_latents = torch.cat([ref_tokens, noisy_target], dim=1)
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

        # --- per-token reference slot ids (ref part precomputed above; target -> padding index) ---
        pad_idx = self.config.max_reference_slots
        target_slot_ids = torch.full((target_seq_len,), pad_idx, dtype=torch.long, device=device)
        reference_slot_ids = torch.cat([ref_slot_ids, target_slot_ids]).unsqueeze(0).expand(batch_size, -1)

        # --- positions: ref part precomputed above (per-image grid + time scheme); target = normal video ---
        target_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width, batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )
        positions = torch.cat([ref_positions, target_positions], dim=2)

        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
            reference_slot_ids=reference_slot_ids,
        )
        # loss on target non-conditioning tokens only (refs excluded)
        video_loss_mask = torch.cat([
            torch.zeros(batch_size, ref_seq_len, dtype=torch.bool, device=device),
            ~target_cond_mask,
        ], dim=1)

        audio_modality = audio_targets = audio_loss_mask = None
        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch, sigmas=sigmas, audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask, batch_size=batch_size, device=device, dtype=dtype,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            ref_seq_len=ref_seq_len,
        )

    def _prepare_audio_inputs(  # noqa: PLR0913
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Joint audio-video: noise the audio with the same sigma, loss on all audio tokens.
        Mirrors TextToVideoStrategy._prepare_audio_inputs."""
        audio_latents = batch["audio_latents"]["latents"]
        audio_latents = self._audio_patchifier.patchify(audio_latents)  # [B, T_a, 128]
        audio_seq_len = audio_latents.shape[1]

        audio_noise = torch.randn_like(audio_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise
        audio_targets = audio_noise - audio_latents
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len, batch_size=batch_size, device=device, dtype=dtype,
        )
        audio_modality = Modality(
            enabled=True, latent=noisy_audio, sigma=sigmas, timesteps=audio_timesteps,
            positions=audio_positions, context=audio_prompt_embeds, context_mask=prompt_attention_mask,
        )
        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)
        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Masked MSE: video on the target slice (refs excluded) + optional audio. Returns [B,]."""
        # video_pred is [B, ref_seq_len + target_seq_len, C]; keep only the target portion.
        target_pred = video_pred[:, inputs.ref_seq_len :, :]
        target_mask = inputs.video_loss_mask[:, inputs.ref_seq_len :].unsqueeze(-1).float()
        video_loss = (target_pred - inputs.video_targets).pow(2).mul(target_mask)
        video_loss = video_loss.mean(dim=[-2, -1]) / target_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss
        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean(dim=[-2, -1])
        return video_loss + audio_loss
