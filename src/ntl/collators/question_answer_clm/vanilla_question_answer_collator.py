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
        text = [f"{example['question']} {example['answer']}{self.tokenizer.eos_token}" for example in examples]
        encodings = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=1024)

        # Create masks for questions in a batch-wise manner
        questions = [example['question'] + " " for example in examples]
        question_encodings = self.tokenizer(questions, padding=False, truncation=True, return_tensors=None)
        question_lengths = torch.tensor([len(enc) - 1 for enc in question_encodings['input_ids']])  # -1 to not count end token
        
        attention_mask = encodings["attention_mask"]
        # Rest of the processing
        input_ids = encodings["input_ids"]
        labels = input_ids.clone()

        # Create position indices tensor
        batch_size, seq_len = labels.size()
        position_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        # Zero out attention mask for all positions less than question length for each batch
        labels[position_indices < question_lengths.unsqueeze(1)] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100

        generation_inputs = [example['question'] for example in examples]
        generation_labels = [f" {example['answer']}{self.tokenizer.eos_token}" for example in examples]
        generation_input = self.tokenizer(generation_inputs, padding=True, truncation=True, return_tensors="pt", padding_side='left', max_length=1000)
        generation_labels = self.tokenizer(generation_labels, padding=True, truncation=True, return_tensors="pt", max_length=1024)

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
