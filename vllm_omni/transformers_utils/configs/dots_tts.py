# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for dots.tts — step 5d.

dots.tts splits its config across two files in the HF checkpoint:
  * ``config.json``      — top-level dots.tts fields (latent_dim / patch_size
                            / PatchEncoder / DiT / vocoder / etc.).
  * ``llm_config.json``  — full Qwen2 LM config (vocab_size / hidden_size /
                            num_hidden_layers / rope_theta / etc.).

``AutoConfig.from_pretrained`` only consumes ``config.json`` automatically, so
this class hoists the Qwen2 fields to top-level attributes with defaults
matching the dots.tts-soar checkpoint.  vLLM's ``Qwen2Model.__init__`` then
reads ``config.hidden_size`` / ``config.vocab_size`` / etc. directly through
``get_text_config()``, the same pattern voxcpm2 follows.

Future work: read ``llm_config.json`` properly during ``from_pretrained`` so
users can swap to other dots.tts variants (mf / base) without editing this
file — for now soar values are hardcoded just like the AudioVAE / DiT /
patch_encoder factories in the talker.

Upstream reference:
  https://huggingface.co/rednote-hilab/dots.tts-soar/blob/main/llm_config.json
"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class DotsTTSConfig(PretrainedConfig):
    """Configuration for dots.tts integration (Qwen2.5-1.5B-Base AR backbone)."""

    model_type = "dots_tts"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # -- top-level dots.tts params --
        architecture: str = "dots_tts",
        lm_config: dict | None = None,
        dit_config: dict | None = None,
        audio_vae_config: dict | None = None,
        bigvgan_config: dict | None = None,
        speaker_encoder_config: dict | None = None,
        sample_rate: int = 48000,
        dtype: str = "bfloat16",
        device: str = "cuda",
        # -- LM defaults (overridden by lm_config dict if present) --
        # All values match dots.tts-soar's llm_config.json.
        vocab_size: int = 151672,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        tie_word_embeddings: bool = True,
        hidden_act: str = "silu",
        bos_token_id: int = 151643,
        eos_token_id: int = 151643,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.architecture = architecture

        # -- dots.tts-specific fields (kept as dicts for now; talker's
        # hardcoded factories already cover the soar values) --
        self.lm_config = lm_config or {}
        self.dit_config = dit_config or {}
        self.audio_vae_config = audio_vae_config or {}
        self.bigvgan_config = bigvgan_config or {}
        self.speaker_encoder_config = speaker_encoder_config or {}
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.device = device

        # -- Hoist LM parameters to top-level for vLLM's Qwen2Model --
        lm = self.lm_config
        self.vocab_size = lm.get("vocab_size", vocab_size)
        self.hidden_size = lm.get("hidden_size", hidden_size)
        self.intermediate_size = lm.get("intermediate_size", intermediate_size)
        self.num_hidden_layers = lm.get("num_hidden_layers", num_hidden_layers)
        self.num_attention_heads = lm.get(
            "num_attention_heads", num_attention_heads
        )
        self.num_key_value_heads = lm.get(
            "num_key_value_heads", num_key_value_heads
        )
        self.max_position_embeddings = lm.get(
            "max_position_embeddings", max_position_embeddings
        )
        self.rms_norm_eps = lm.get("rms_norm_eps", rms_norm_eps)
        self.rope_theta = lm.get("rope_theta", rope_theta)
        self.hidden_act = lm.get("hidden_act", hidden_act)
        self.head_dim = self.hidden_size // self.num_attention_heads

        # vLLM's Qwen2Model reads ``max_window_layers`` and checks against
        # ``num_hidden_layers`` only when ``use_sliding_window`` is true.
        # soar disables sliding window — set both to the same value so the
        # assertion (if executed) stays vacuous.
        self.use_sliding_window = lm.get("use_sliding_window", False)
        self.sliding_window = lm.get("sliding_window", None)
        self.max_window_layers = lm.get(
            "max_window_layers", self.num_hidden_layers
        )

    def get_text_config(self, **kwargs):
        """Return self — LM fields are hoisted to top-level attributes."""
        return self


AutoConfig.register("dots_tts", DotsTTSConfig)

__all__ = ["DotsTTSConfig"]
