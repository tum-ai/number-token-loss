from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class T5RegressionModelXval(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer, dim_feedforward=1024, numhead_bias=True):
        super().__init__(config)
        super()._resize_token_embeddings(config.vocab_size)
        
        self.tokenizer = tokenizer

        self.num_head = nn.Sequential(
            nn.Linear(config.d_model, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_number_embeddings: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            number_labels: Optional[torch.LongTensor] = None,
            **kwargs
            
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """Overwrites forward method of parent class T5ForConditionalGeneration. Computes embeddings
        from input_ids and stores them to inputs_embeds. Argument inputs_embeds will not persist.
        """
        # create embeddings
        inputs_embeds = self.shared(input_ids) * input_number_embeddings.unsqueeze(-1)

        # call super().forward()
        outputs = super(T5RegressionModelXval, self).forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        sequence_output = outputs.decoder_hidden_states[-1]
        loss_mlm = outputs.loss

        num_preds = self.num_head(sequence_output)

        num_mask = labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.num_token)

        loss_num = F.mse_loss(
            num_preds[num_mask],
            number_labels[num_mask].view(-1, 1),
            reduction="mean",
        )
        loss = loss_mlm + loss_num

        outputs.loss = loss
        outputs["number_predictions"] = num_preds

        return outputs
