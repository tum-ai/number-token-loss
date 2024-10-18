from typing import List, Union, Dict

import torch
from transformers import DataCollatorForLanguageModeling

from src.utils.numerical_operations import signed_log


class XvalMaskedQuestionAnswerCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        # Tokenize questions and answers separately
        questions = [example['question'] for example in examples]
        answers = [example['answer'] for example in examples]

        text = [question + " " + answer for question, answer in zip(questions, answers)]

        encodings = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        x = encodings['input_ids']
        x_num = encodings["number_embeddings"]
        # attention_mask = question_encodings['attention_mask']

        # mask for last number token
        number_token_id = self.tokenizer.get_num_token_ids()[0]

        # mask, which is only true for the last number token
        number_token_mask = x == number_token_id

        x_num[number_token_mask] = signed_log(x_num[number_token_mask])

        batch_size, seq_len = x.size()
        indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        number_indices = indices.masked_fill(~number_token_mask, -1)
        last_number_token_idx = number_indices.max(dim=1)[0]
        last_number_token_mask = torch.zeros_like(x, dtype=torch.bool)
        valid = last_number_token_idx >= 0
        batch_indices = torch.arange(batch_size)[valid]
        token_indices = last_number_token_idx[valid]
        last_number_token_mask[batch_indices, token_indices] = True

        y = x.clone()
        y_num = x_num.clone()
        y[~last_number_token_mask] = -100
        x[last_number_token_mask] = self.tokenizer.additional_special_tokens_ids[0]
        x_num[last_number_token_mask] = 1

        mask = last_number_token_mask

        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask}