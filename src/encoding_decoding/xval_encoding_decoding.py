import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class XValModel(nn.Module):
    def __init__(self, transformer_backbone, vocab_size, dim_feedforward=3072, context_length=190,
                 transformer_bias=False, numhead_bias=True):
        super(XValModel, self).__init__()
        self.transformer_backbone = transformer_backbone

        self.token_embed = nn.Embedding(vocab_size, self.transformer_backbone.hidden_size)
        self.position_embed = nn.Embedding(context_length, self.transformer_backbone.hidden_size)

        self.lm_head = nn.Sequential(
            nn.Linear(self.transformer_backbone.hidden_size, dim_feedforward, bias=transformer_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, vocab_size, bias=transformer_bias),
        )
        self.num_head = nn.Sequential(
            nn.Linear(self.transformer_backbone.hidden_size, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )

    def forward(self, x, x_num, attention_mask, token_type_ids):
        token_embeddings = self.token_embed(x) * x_num.unsqueeze(-1)
        position_embeddings = self.position_embed.weight[: x.shape[1]].unsqueeze(0)
        embeddings = token_embeddings + position_embeddings
        sequence_output = self.transformer_backbone(embeddings=embeddings, attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)
        logit_preds = self.lm_head(sequence_output)
        num_preds = self.num_head(sequence_output)
        return logit_preds, num_preds


def define_masked_num_collator(pad_token_id, mask_token_id, mlm_probability):
    def masked_num_collator(batch):
        x = [torch.tensor(sample["input_ids"]) for sample in batch]
        x_num = [torch.tensor(sample["numbers"]) for sample in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
        x_num = pad_sequence(x_num, batch_first=True, padding_value=1)
        attention_mask = (x != pad_token_id).long()
        token_type_ids = torch.zeros_like(x)  # XLNet uses token type ids
        probability_matrix = torch.full(x.shape, mlm_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        y = x.clone()
        y_num = x_num.clone()
        y[~mask] = -100
        x[mask] = mask_token_id
        x_num[mask] = 1
        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask, "attention_mask": attention_mask,
                "token_type_ids": token_type_ids}

    return masked_num_collator
