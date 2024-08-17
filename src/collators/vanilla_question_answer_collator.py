from typing import Dict, List, Union

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number



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

        # Generate number_labels for easier evaluation
        """        number_token_ids = self.tokenizer.get_num_token_ids()
        label_num_mask = torch.isin(answer_input_ids, torch.tensor(number_token_ids, dtype=torch.long, device=answer_input_ids.device))
        tokens = self.tokenizer.convert_ids_to_tokens(answer_input_ids[label_num_mask])
        number_values = torch.tensor([encoding_to_number(token) for token in tokens], dtype=torch.float, device=answer_input_ids.device)
        number_labels = torch.zeros_like(answer_input_ids, dtype=torch.float)
        number_labels[label_num_mask] = number_values"""

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
