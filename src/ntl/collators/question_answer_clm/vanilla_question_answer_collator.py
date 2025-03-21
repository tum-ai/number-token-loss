from typing import Dict, List, Union

import torch
from transformers import DataCollatorForLanguageModeling

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class VanillaQuestionAnswerCLMCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: NumberEncodingTokenizer):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer: NumberEncodingTokenizer = tokenizer
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
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
