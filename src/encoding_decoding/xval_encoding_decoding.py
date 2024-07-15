import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

MODEL_TO_EMBEDDING_FN = {
    "XLNetBackbone": "self.transformer_backbone.transformer.word_embedding",
    "T5RegressionModel": "self.transformer_backbone.model.shared",
}

class XValModel(nn.Module):
    def __init__(self, transformer_backbone, vocab_size, dim_feedforward=3072, context_length=190,
                 transformer_bias=False, numhead_bias=True):
        super(XValModel, self).__init__()
        self.transformer_backbone = transformer_backbone

        self.transformer_backbone.transformer.resize_token_embeddings(vocab_size)
        model_name = transformer_backbone._get_name()
        self.embed_fn = eval(MODEL_TO_EMBEDDING_FN[model_name])

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
        token_embeddings = self.embed_fn(x) * x_num.unsqueeze(-1)
        sequence_output = self.transformer_backbone(embeddings=token_embeddings, attention_mask=attention_mask,
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
