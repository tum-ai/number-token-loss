from typing import Optional, Tuple, Union
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
from src.encoding_decoding.numerical_encodings import FloatEncoding
import copy

# Set maximal value for normalization
V_MAX = 3000000000


class T5RegressionModelRT(T5ForConditionalGeneration):
    _tied_weights_keys = [
        "encoder.embed_tokens.token_embeddings.weight",
        "encoder.embed_tokens.number_embeddings.weight",
        'encoder.embed_tokens.number_embeddings.embedding.weight',
        "decoder.embed_tokens.token_embeddings.weight",
        "decoder.embed_tokens.number_embeddings.weight",
        "decoder.embed_tokens.number_embeddings.embedding.weight",
        "lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        super()._resize_token_embeddings(config.vocab_size)
        number_embeds = FloatEncoding(num_embeddings=config.vocab_size, embedding_dim=self.config.d_model,
                                      vocab=config.added_vocab, vmax=V_MAX)
        combined_embeddings = RTEmbeddings(self.shared, number_embeds)
        # Set the new embedding for encoder and decoder.
        self.set_input_embeddings(combined_embeddings)

    def forward(
            self,
            labels: Optional[torch.LongTensor] = None,
            number_labels: Optional[torch.LongTensor] = None,
            **kwargs,

    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """Overwrites forward method of parent class T5ForConditionalGeneration. Computes embeddings
        from input_ids and stores them to inputs_embeds. Argument inputs_embeds will not persist.
        """
        # call super().forward()
        outputs = super(T5RegressionModelRT, self).forward(labels=labels, **kwargs)
        return outputs


class RTEmbeddings(nn.Module):
    def __init__(self, token_embeddings, number_embeddings):
        super().__init__()
        self.token_embeddings = copy.deepcopy(token_embeddings)
        self.number_embeddings = number_embeddings

    def forward(self, input_ids):
        # Compute token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Compute number embeddings
        number_embeds = self.number_embeddings(input_ids)

        # Combine embeddings by addition
        combined_embeds = token_embeds + number_embeds
        return combined_embeds
