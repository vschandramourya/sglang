"""
Extends SGLang's base ModelConfig to handle custom Phoenix model architectures
by mapping them to compatible Eagle implementations for draft models.

Base SGLang doesn't support Phoenix architecture names,
this provides compatibility layer for private model variants.
"""

import json
import logging
import os

from sglang.srt.configs.model_config import ModelConfig as SGLModelConfig
from sglang.srt.utils import retry

logger = logging.getLogger(__name__)


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

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        if quant_cfg is None:
            # check if is modelopt or mixed-precision model -- Both of them don't have corresponding field
            # in hf `config.json` but has a standalone `hf_quant_config.json` in the root directory
            # example: https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8/tree/main
            # example: https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/tree/main
            is_local = os.path.exists(self.model_path)
            if not is_local:
                import huggingface_hub

                # Skip HF check if offline mode is enabled
                if huggingface_hub.constants.HF_HUB_OFFLINE:
                    logger.warning(
                        "Offline mode is enabled, skipping hf_quant_config.json check"
                    )
                    pass
                else:
                    try:
                        from huggingface_hub import HfApi, hf_hub_download

                        hf_api = HfApi()
                        # Retry HF API call up to 3 times
                        file_exists = retry(
                            lambda: hf_api.file_exists(
                                self.model_path, "hf_quant_config.json"
                            ),
                            max_retry=2,
                            initial_delay=1.0,
                            max_delay=5.0,
                        )
                        if file_exists:
                            # Download and parse the quantization config for remote models
                            quant_config_file = hf_hub_download(
                                repo_id=self.model_path,
                                filename="hf_quant_config.json",
                                revision=self.revision,
                            )
                            with open(quant_config_file) as f:
                                quant_config_dict = json.load(f)
                            quant_cfg = self._parse_modelopt_quant_config(
                                quant_config_dict
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to check hf_quant_config.json: {self.model_path} {e}"
                        )
            elif os.path.exists(os.path.join(self.model_path, "hf_quant_config.json")):
                quant_config_file = os.path.join(
                    self.model_path, "hf_quant_config.json"
                )
                with open(quant_config_file) as f:
                    quant_config_dict = json.load(f)
                quant_cfg = self._parse_modelopt_quant_config(quant_config_dict)
        return quant_cfg
