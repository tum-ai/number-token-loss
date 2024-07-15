import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

MODEL_TO_EMBEDDING_FN = {
    "XLNetBackbone": "self.transformer_backbone.transformer.word_embedding",
    "T5RegressionModel": "self.transformer_backbone.model.shared",
}


class RegressionTransformer(nn.Module):
    def __init__(self, transformer_backbone, vocab_size, dim_feedforward=3072, context_length=190,
                 transformer_bias=False):
        super(RegressionTransformer, self).__init__()
        self.transformer_backbone = transformer_backbone

        self.transformer_backbone.transformer.resize_token_embeddings(vocab_size)
        model_name = transformer_backbone._get_name()
        self.embed_fn = eval(MODEL_TO_EMBEDDING_FN[model_name])

        self.lm_head = nn.Sequential(
            nn.Linear(self.transformer_backbone.hidden_size, dim_feedforward, bias=transformer_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, vocab_size, bias=transformer_bias),
        )

    def forward(self, x, number_embeddings, attention_mask, token_type_ids):
        token_embeddings = self.embed_fn(x) + number_embeddings  #To do: do we want to overwrite or sum? 
        sequence_output = self.transformer_backbone(embeddings=token_embeddings, attention_mask=attention_mask,
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