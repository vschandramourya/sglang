"""
Extends SGLang's base ModelConfig to handle custom Phoenix model architectures
by mapping them to compatible Eagle implementations for draft models.

Base SGLang doesn't support Phoenix architecture names,
this provides compatibility layer for private model variants.
"""

from sglang.srt.configs.model_config import ModelConfig as SGLModelConfig


class ModelConfig(SGLModelConfig):
    """Extended ModelConfig with Phoenix->Eagle architecture mapping for draft models."""

    def _config_draft_model(self):
        super()._config_draft_model()

        is_draft_model = self.is_draft_model

        # Map Phoenix architecture to Eagle for draft model compatibility
        if (
            is_draft_model
            and self.hf_config.architectures[0] == "PhoenixLlamaForCausalLM"
        ):
            self.hf_config.architectures[0] = "LlamaForCausalLMEagle"
