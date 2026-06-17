# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots.tts talker — step 3b-2.

Wired so far:
  * Qwen2.5-1.5B base LM (vLLM-native, forward returns hidden states).
  * AudioVAE side module (random-init; checkpoint loading lands in step 3c
    when load_weights ingests ``vocoder.safetensors``).

Still stubbed for later steps: DiT flow-matching head, CAM++ speaker encoder,
per-request state, runtime config, CUDA Graph caches, preprocess / make_omni_output
/ compute_logits.  See dots_tts_notes.md §5.1 for the roadmap.

Reference: vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.dots_tts.dots_tts_vocoder import (
    AudioVAE,
    AudioVAEConfig,
)

logger = init_logger(__name__)


# Vocoder hyperparameters from rednote-hilab/dots.tts-soar config.json
# (the ``vocoder`` block).  dots.tts-mf ships identical values.  Step 3c
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


class DotsTTSForConditionalGeneration(nn.Module):
    """dots.tts AR talker.  Step 3b-2: base LM + random-init AudioVAE."""

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

        logger.info(
            "DotsTTS step-3b-2 loaded (base_lm=Qwen2, audio_vae=AudioVAE[random], "
            "model=%s); DiT head / speaker encoder not yet wired.",
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

    def load_weights(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("step 3")
