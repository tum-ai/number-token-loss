from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
from typing import Optional, Tuple, Union
from ntl.loss_functions.number_token_loss import NumberTokenLoss
from ntl.loss_functions.expression_loss import ExpressionLoss


class T5VanillaForNumberTokenLoss(T5ForConditionalGeneration):

    def __init__(
        self,
        config,
        number_token_loss: NumberTokenLoss = None,
        expression_loss: ExpressionLoss = None,
    ):
        super().__init__(config)
        super().resize_token_embeddings(config.vocab_size, pad_to_multiple_of=64 if torch.cuda.is_available() else 1)

        # Initialize additional losses
        self.number_token_loss = number_token_loss
        self.expression_loss = expression_loss

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

        # If labels are provided, calculate and combine the NumberTokenLoss
        if labels is not None and self.number_token_loss is not None:
            number_token_loss = self.number_token_loss.forward(outputs.logits, labels)
            outputs["number_loss"] = number_token_loss
            outputs["token_loss"] = outputs.loss
            outputs.loss = outputs.loss + self.number_token_loss.weight * number_token_loss

        if labels is not None and self.expression_loss is not None:
            expression_loss = self.expression_loss.forward(outputs.logits, labels)
            outputs["expression_loss"] = expression_loss
            outputs.loss = outputs.loss + self.expression_loss.weight * expression_loss
        return outputs
