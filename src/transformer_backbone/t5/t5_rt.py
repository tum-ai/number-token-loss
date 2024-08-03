from typing import Optional, Tuple, Union
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
from src.encoding_decoding.numerical_encodings import FloatEncoding


class T5RegressionModelRT(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        super()._resize_token_embeddings(config.vocab_size)
        self.set_number_embeds(num_embeddings=config.vocab_size, vocab=config.added_vocab)

    def set_number_embeds(self, num_embeddings, vocab):
        self.number_embeds = FloatEncoding(num_embeddings=num_embeddings, embedding_dim=self.config.d_model,
                                           vocab=vocab)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            number_labels: Optional[torch.LongTensor] = None,
            **kwargs,

    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """Overwrites forward method of parent class T5ForConditionalGeneration. Computes embeddings
        from input_ids and stores them to inputs_embeds. Argument inputs_embeds will not persist.
        """
        # create embeddings
        inputs_embeds = self.shared(input_ids)

        # compute number embeddings and add them to the output
        number_embeds = self.number_embeds(input_ids)
        inputs_embeds += number_embeds

        # call super().forward()
        outputs = super(T5RegressionModelRT, self).forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs
