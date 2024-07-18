from typing import Dict, List, Union
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling


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



class CustomMaskingCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        # Tokenize questions and answers separately
        questions = [example['question'] for example in examples]
        answers = [example['answer'] for example in examples]

        question_encodings = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
        answer_encodings = self.tokenizer(answers, padding=True, truncation=True, return_tensors="pt")

        input_ids = question_encodings['input_ids']
        attention_mask = question_encodings['attention_mask']

        # Masking the answers
        answer_input_ids = answer_encodings['input_ids']
        labels = answer_input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }