import logging

import flax
import flax.linen as nn
import jax
import numpy as np

class LanguageEncoder(nn.Module):
    """
    Language encoder that embeds text input IDs into continuous language embeddings. Supports pre-trained HF models.

     Args:
         num_tokens (int): Number of output tokens (not enforced).
         encoder (str, optional): Optional HuggingFace AutoModel name for encoding input IDs.
         finetune_encoder (bool, optional): Optional finetune last layers of the language model.
    """

    encoder: str = None
    finetune_encoder: bool = False

    def setup(self):
        if self.encoder is not None:
            from transformers import AutoConfig, FlaxAutoModel, FlaxT5EncoderModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

            config = AutoConfig.from_pretrained(self.encoder)
            if "t5" in self.encoder:
                self.hf_model = FlaxT5EncoderModel(config).module
            else:
                self.hf_model = FlaxAutoModel.from_config(config).module

    def __call__(
        self,
        tasks=None,
    ):
        if "language_instruction" not in tasks:
            logging.warning("No language inputs found. Skipping tokenizer entirely.")
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None
        else:
            tokens = self.tokenizer(tasks["language_instruction"], return_tensors="np")
            tokens = self.hf_model(tokens).last_hidden_state

        if not self.finetune_encoder:
            tokens = jax.lax.stop_gradient(tokens)

        return tokens