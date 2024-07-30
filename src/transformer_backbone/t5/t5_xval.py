from typing import Optional, Tuple, Union
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoding_decoding.numerical_encodings import FloatEncoding


class T5RegressionModelXval(T5ForConditionalGeneration):
    def __init__(self, config, dim_feedforward=1024, numhead_bias=True):
        super().__init__(config)

        self.num_head = nn.Sequential(
            nn.Linear(self.transformer_backbone.hidden_size, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_number_embeddings: Optional[torch.LongTensor] = None,
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
        """Overwrites forward method of parent class T5ForConditionalGeneration. Computes embeddings
        from input_ids and stores them to inputs_embeds. Argument inputs_embeds will not persist.
        """
        # create embeddings
        inputs_embeds = self.shared(input_ids) * input_number_embeddings

        # call super().forward()
        outputs = super(T5RegressionModelXval, self).forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        sequence_output = outputs.decoder_hidden_states[-1]
        loss_mlm = outputs.loss

        num_preds = self.num_head(sequence_output)

        num_mask = labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.num_token)

        loss_num = F.mse_loss(
            num_preds[num_mask],
            number_labels[num_mask].view(-1, 1).cuda(),
            reduction="mean",
        )
        loss = loss_mlm + loss_num

        outputs.loss = loss

        return outputs
