from typing import Optional, Tuple, Union
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
from src.encoding_decoding.numerical_encodings import FloatEncoding

class T5RegressionModel(T5ForConditionalGeneration):
    def __init__(self, config, num_output=1):
        super().__init__(config)
        self.regression_head = nn.Linear(self.config.d_model, num_output)

    def set_number_embeds(self, num_embeddings, vocab):
        self.number_embeds = FloatEncoding(num_embeddings=num_embeddings, embedding_dim=self.config.d_model, vocab=vocab)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        outputs = super(T5RegressionModel, self).forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )        

        return outputs
    
        """def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        # hidden state of the last token is output
        sequence_output = hidden_states[:, -1, :]

        return sequence_output"""
