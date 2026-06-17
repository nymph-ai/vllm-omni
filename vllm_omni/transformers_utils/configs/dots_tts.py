# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for dots.tts (V1 skeleton).

仅声明 model_type='dots_tts',让 AutoConfig.from_pretrained 不报错。
真正字段(Qwen2.5 LM 参数、DiTAR/MeanFlow 头超参、AudioVAE 通道数、BigVGAN
config、CAM++ x-vector 维度)等读到 dots.tts 上游 checkpoint 的 config.json
后补齐。

上游参考:https://huggingface.co/rednote-hilab/dots.tts.soar
"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class DotsTTSConfig(PretrainedConfig):
    """Configuration for dots.tts integration.

    dots.tts 用 Qwen2.5-1.5B 作为 AR backbone(vLLM 原生支持),所以 LM
    部分的字段大概率从 lm_config 字典里拿,不重新写一遍。Talker 侧路
    (DiTAR/MeanFlow head / AudioVAE / BigVGAN / CAM++)的字段等真实
    config.json 读到再加。
    """

    model_type = "dots_tts"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        architecture: str = "dots_tts",
        lm_config: dict | None = None,
        # talker 侧路配置(占位)
        dit_config: dict | None = None,
        audio_vae_config: dict | None = None,
        bigvgan_config: dict | None = None,
        speaker_encoder_config: dict | None = None,
        # 输出
        sample_rate: int = 48000,
        dtype: str = "bfloat16",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.architecture = architecture
        self.lm_config = lm_config or {}
        self.dit_config = dit_config or {}
        self.audio_vae_config = audio_vae_config or {}
        self.bigvgan_config = bigvgan_config or {}
        self.speaker_encoder_config = speaker_encoder_config or {}
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.device = device

    def get_text_config(self, **kwargs):
        """Return self as the text config (LM attributes will hoist when wired up)."""
        return self


AutoConfig.register("dots_tts", DotsTTSConfig)

__all__ = ["DotsTTSConfig"]
