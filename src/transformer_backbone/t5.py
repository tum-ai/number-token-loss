from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
import torch
import torch.nn as nn

class T5RegressionModel(nn.Module):
    def __init__(self, pretrained_model_name='t5-small', num_output=1):
        super(T5RegressionModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.regression_head = nn.Linear(self.model.config.d_model, num_output)

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        # hidden state of the last token is output
        sequence_output = hidden_states[:, -1, :]

        return sequence_output
