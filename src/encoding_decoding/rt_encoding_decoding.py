import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class RegressionTransformer(nn.Module):
    def __init__(self, transformer_backbone, vocab_size, dim_feedforward=3072, context_length=190,
                 transformer_bias=False):
        super(RegressionTransformer, self).__init__()
        self.transformer_backbone = transformer_backbone

        self.token_embed = nn.Embedding(vocab_size, self.transformer_backbone.hidden_size)
        self.position_embed = nn.Embedding(context_length, self.transformer_backbone.hidden_size)

        self.lm_head = nn.Sequential(
            nn.Linear(self.transformer_backbone.hidden_size, dim_feedforward, bias=transformer_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, vocab_size, bias=transformer_bias),
        )

    def forward(self, x, number_embeddings, attention_mask, token_type_ids):
        token_embeddings = self.token_embed(x) + number_embeddings
        position_embeddings = self.position_embed.weight[:x.shape[1]].unsqueeze(0)

        embeddings = token_embeddings + position_embeddings
        sequence_output = self.transformer_backbone(embeddings=embeddings, attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)
        logit_preds = self.lm_head(sequence_output)
        return logit_preds


def define_masked_num_collator(pad_token_id, mask_token_id, mlm_probability):
    def masked_num_collator(batch):
        x = [torch.tensor(sample["input_ids"]) for sample in batch]
        number_embeddings = [torch.tensor(sample["number_embeddings"]) for sample in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
        number_embeddings = pad_sequence(number_embeddings, batch_first=True, padding_value=0)
        attention_mask = (x != pad_token_id).long()
        token_type_ids = torch.zeros_like(x)  # XLNet uses token type ids
        probability_matrix = torch.full(x.shape, mlm_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        y = x.clone()
        y_num = number_embeddings.clone()
        y[~mask] = -100
        x[mask] = mask_token_id
        number_embeddings[mask] = 0
        return {"x": x, "number_embeddings": number_embeddings, "y": y, "y_num": y_num, "mask": mask, "attention_mask": attention_mask,
                "token_type_ids": token_type_ids}

    return masked_num_collator