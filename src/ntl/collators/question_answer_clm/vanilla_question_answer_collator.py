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
        # Tokenize questions and answers as a single sequence
        text = [f"{example['question']} {example['answer']}|||||||" for example in examples]
        encodings = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        generation_inputs = [example['question'] for example in examples]
        generation_labels = [f"{example['answer']}|" for example in examples]
        generation_input = self.tokenizer(generation_inputs, padding=True, truncation=True, return_tensors="pt", padding_side='left')
        generation_labels = self.tokenizer(generation_labels, padding=True, truncation=True, return_tensors="pt")

        # input_ids = torch.cat([generation_input["input_ids"], generation_labels["input_ids"]], dim=1)
        # attention_mask = torch.cat([generation_input["attention_mask"], generation_labels["attention_mask"]], dim=1)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Masking the answers
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        generation_input_ids = generation_input["input_ids"]
        generation_attention_mask = generation_input["attention_mask"]

        generation_label_ids = generation_labels["input_ids"]
        generation_label_ids[generation_label_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'generation_input_ids': generation_input_ids,
            'generation_attention_mask': generation_attention_mask,
            'generation_labels': generation_label_ids
        }
