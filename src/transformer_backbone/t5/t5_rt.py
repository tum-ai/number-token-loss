from typing import Optional, Tuple, Union
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
from src.encoding_decoding.numerical_encodings import FloatEncoding
import copy
from src.number_token_loss import NumberTokenLoss

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

    def __init__(self, config, number_token_loss: NumberTokenLoss = None):
        super().__init__(config)
        super()._resize_token_embeddings(config.vocab_size)
        number_embeds = FloatEncoding(num_embeddings=config.vocab_size, embedding_dim=self.config.d_model,
                                      vocab=config.added_vocab, vmax=V_MAX)
        combined_embeddings = RTEmbeddings(self.shared, number_embeds)
        # Set the new embedding for encoder and decoder.
        self.set_input_embeddings(combined_embeddings)
        self.number_token_loss = number_token_loss


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
            number_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        # Call the parent's forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # If labels are provided, calculate the NumberTokenLoss
        if labels is not None and self.number_token_loss is not None:
            number_token_loss = self.number_token_loss.forward(outputs.logits, labels)
            # number_token_loss = torch.log10(number_token_loss + 1)
            outputs["number_loss"] = number_token_loss
            outputs["token_loss"] = outputs.loss
            outputs.loss = (1.0 - self.number_token_loss.weight) * outputs.loss + \
                        self.number_token_loss.weight * number_token_loss
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
