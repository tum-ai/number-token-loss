from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
import torch
import torch.nn as nn


class T5RegressionModel(nn.Module):
    def __init__(self, embedding_dim: int, input_length, pretrained_model_name='t5-small', num_output=1):
        super(T5RegressionModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.model.resize_token_embeddings(embedding_dim)
        self.model.resize_position_embeddings(input_length)

        self.hidden_size = self.transformer.config.hidden_size

    def pretrained_embed(self, input_ids):
        positional_embeddings = self.model.get_position_embeddings()(input_ids)
        word_embeddings = self.model.get_output_embeddings()(input_ids)
        return positional_embeddings, word_embeddings

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        # hidden state of the last token is output
        sequence_output = hidden_states[:, -1, :]

        return sequence_output
