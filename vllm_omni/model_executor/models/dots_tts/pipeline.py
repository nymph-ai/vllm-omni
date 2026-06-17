# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots.tts pipeline topology (frozen).

Single-stage AR TTS: text → 48 kHz waveform in one pass.
Architecture: Qwen2.5-1.5B base LM (vLLM-native) → DiTAR / MeanFlow head
→ AudioVAE.decode → BigVGAN → 48 kHz mono.

Speaker conditioning via CAM++ x-vector injected in preprocess().

Modeled after voxcpm2's single-stage pipeline (same "vLLM-native base LM
+ side-path computation" pattern), but simplified: no residual LM, no FSQ.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

DOTS_TTS_PIPELINE = PipelineConfig(
    model_type="dots_tts",
    model_arch="DotsTTSForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="latent_generator",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                # Qwen2.5 EOS = 151643; refine once real checkpoint loads.
                "stop_token_ids": [151643],
            },
        ),
    ),
)
