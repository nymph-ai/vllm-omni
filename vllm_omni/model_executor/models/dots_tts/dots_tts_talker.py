# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots.tts talker — step 4c.

Wired so far:
  * Qwen2.5-1.5B base LM (vLLM-native, forward returns hidden states).
  * AudioVAE side module + real weights via ``vocoder.safetensors``.
  * DiT flow-matching head + real weights via ``model.safetensors``'s
    ``velocity_field_predictor.*`` namespace.

Still stubbed for later steps: CAM++ speaker encoder, patch_encoder /
projectors / eos_proj, per-request state, runtime config, CUDA Graph caches,
preprocess / make_omni_output / compute_logits.  See dots_tts_notes.md §5.1
for the roadmap.

Reference: vllm_omni/model_executor/models/ming_flash_omni/ming_flash_omni_talker.py
(its ``_load_vae_weights`` is the model for our AudioVAE branch).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.dots_tts.dots_tts_dit import DiT
from vllm_omni.model_executor.models.dots_tts.dots_tts_vocoder import (
    AudioVAE,
    AudioVAEConfig,
)

logger = init_logger(__name__)


# Vocoder hyperparameters from rednote-hilab/dots.tts-soar config.json
# (the ``vocoder`` block).  dots.tts-mf ships identical values.  A future step
# will replace this hard-coded factory with a parse off the real HF config.
def _build_soar_audio_vae_config() -> AudioVAEConfig:
    return AudioVAEConfig(
        sample_rate=48000,
        upsample_rates=[10, 6, 4, 2, 2, 2],
        upsample_kernel_sizes=[20, 12, 8, 4, 4, 4],
        upsample_initial_channel=1536,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        downsample_rates=[2, 2, 2, 4, 6, 10],
        downsample_channels=[12, 24, 48, 96, 192, 384, 768],
        activation="snakebeta",
        snake_logscale=True,
        latent_dim=128,
        causal=True,
        mi_num_layers=4,
        causal_encoder=True,
        use_bias_at_final=False,
        use_tanh_at_final=False,
    )


# DiT transformer hyperparameters from rednote-hilab/dots.tts-soar config.json
# (the ``DiT`` block).  Mirrors the shape of an upstream pydantic config: DiT's
# constructor calls ``transformer_config.to_dict()`` and reads ``.hidden_size``
# / ``.num_layers``, so this @dataclass plus the ``to_dict`` helper satisfies
# that minimal interface without dragging in upstream's config system.
#
# Note: ``attn_dropout`` in the JSON spells differently from MultiHeadAttention's
# ``attn_drop`` kwarg — the value gets absorbed by ``**_kwargs`` and the MHA
# default 0.0 applies.  Same numeric outcome for the soar checkpoint (both 0.0),
# so this is benign; flagging it because it would mask non-default values if
# upstream ever raised this field.
@dataclass
class _DiTConfig:
    num_layers: int = 18
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    modulation: bool = True
    qkv_bias: bool = False
    qk_norm: bool = True
    attn_dropout: float = 0.0
    dropout: float = 0.0
    norm_layer: str = "RMSNorm"
    alibi_bias: bool = False
    rotary_bias: bool = True
    rotary_theta: float = 10000.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build_soar_dit_config() -> _DiTConfig:
    return _DiTConfig()  # all defaults already match dots.tts-soar


class DotsTTSForConditionalGeneration(nn.Module):
    """dots.tts AR talker.  Step 4c: base LM + AudioVAE + DiT head (real weights)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.model = Qwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self._audio_vae = AudioVAE(_build_soar_audio_vae_config())
        # Upstream serializes vocoder.safetensors with weight_norm folded into
        # plain `weight` on the decoder (encoder kept weight_norm — it's not in
        # the synthesis hot path).  Match that layout before load_weights runs,
        # so checkpoint keys align with our state_dict 1:1.
        self._audio_vae.remove_weight_norm()

        # DiT flow-matching head.  Dimensions per upstream core.py:101-104:
        #   in_dim  = config.DiT.hidden_size = 1024  (DiT internal space)
        #   out_dim = config.latent_dim      = 128   (AudioVAE input space)
        # mode is "flow_matching" for soar/base; "meanflow" only for the mf
        # checkpoint (handled by a separate factory in a later step).
        self._head = DiT(
            in_dim=1024,
            out_dim=128,
            transformer_config=_build_soar_dit_config(),
            mode="flow_matching",
        )

        logger.info(
            "DotsTTS step-4c built (base_lm=Qwen2, audio_vae=AudioVAE[random until "
            "load_weights], dit=DiT[18L,16H,1024d,random until load_weights], "
            "model=%s); speaker encoder / patch_encoder not yet wired.",
            vllm_config.model_config.model,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(output, IntermediateTensors):
            return output
        if isinstance(output, tuple):
            output = output[0]
        return output

    # ── vllm-omni contract stubs (real impls land in step 3+) ──

    def preprocess(self, *args: Any, **kwargs: Any) -> tuple:
        raise NotImplementedError("step 4")

    def postprocess(self, *args: Any, **kwargs: Any) -> dict:
        return {}

    def make_omni_output(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("step 3")

    def compute_logits(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("step 3")

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load AudioVAE + DiT weights — step 4c.

        Two routing branches in a single pass over the input iterator (it can
        only be consumed once):

        * **AudioVAE**: keys from ``vocoder.safetensors``.  Upstream serializes
          this file stand-alone (no parent prefix), but we still strip a leading
          ``vocoder.`` defensively in case a future export wraps it.
        * **DiT**: keys from ``model.safetensors`` under the exact prefix
          ``velocity_field_predictor.`` (confirmed against the dots.tts-soar
          checkpoint header — see /tmp/diff_dit_keys.py).

        Every other key (``llm.*``, ``patch_encoder.*``, ``eos_proj.*``, the
        various projectors) is silently ignored; later steps will pick those
        up by adding new branches alongside these two.
        """
        vae_state_keys = set(self._audio_vae.state_dict().keys())
        dit_state_keys = set(self._head.state_dict().keys())

        matched_vae: list[tuple[str, torch.Tensor]] = []
        matched_dit: list[tuple[str, torch.Tensor]] = []

        DIT_PREFIX = "velocity_field_predictor."
        for name, tensor in weights:
            if name.startswith(DIT_PREFIX):
                candidate = name[len(DIT_PREFIX):]
                if candidate in dit_state_keys:
                    matched_dit.append((candidate, tensor))
                continue
            candidate = (
                name[len("vocoder."):] if name.startswith("vocoder.") else name
            )
            if candidate in vae_state_keys:
                matched_vae.append((candidate, tensor))

        loaded: set[str] = set()

        if matched_vae:
            vae_loader = AutoWeightsLoader(self._audio_vae)
            loaded_vae = vae_loader.load_weights(iter(matched_vae))
            loaded.update(f"_audio_vae.{name}" for name in loaded_vae)
            logger.info(
                "DotsTTS step-4c: loaded %d/%d AudioVAE tensors.",
                len(loaded_vae),
                len(vae_state_keys),
            )

        if matched_dit:
            dit_loader = AutoWeightsLoader(self._head)
            loaded_dit = dit_loader.load_weights(iter(matched_dit))
            loaded.update(f"_head.{name}" for name in loaded_dit)
            logger.info(
                "DotsTTS step-4c: loaded %d/%d DiT tensors.",
                len(loaded_dit),
                len(dit_state_keys),
            )

        if not matched_vae and not matched_dit:
            logger.warning(
                "DotsTTS load_weights step-4c: no AudioVAE or DiT keys matched."
            )

        return loaded
