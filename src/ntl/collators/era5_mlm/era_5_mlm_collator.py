from typing import Dict, List, Union

import torch
from transformers import DataCollatorForLanguageModeling

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class Era5MLMCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: NumberEncodingTokenizer):
        # Set mlm=True for masked language modeling
        super().__init__(tokenizer, mlm=False)
        self.tokenizer: NumberEncodingTokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token = self.tokenizer.additional_special_tokens[0]  # Assuming first special token is the mask token

    def __call__(self, examples: List[Dict[str, Union[List, Dict]]]) -> Dict[str, torch.Tensor]:
        # Process each example to create masked inputs and labels
        masked_inputs = []
        labels = []
        
        for example in examples:
            # Get the data lists
            data_lists = example['data']
            
            # Create input by masking the last element of each list
            label_data = []
            
            for idx, data_list in enumerate(data_lists):
                # Store the last value for labels
                last_value = data_list[-1]
                # Replace last value with mask token
                data_list[-1] = self.tokenizer.additional_special_tokens[idx]

                label_data.append(f"{self.tokenizer.additional_special_tokens[idx]} {last_value}")
            
            # Convert numerical data to strings for tokenization
            masked_text = str(example).replace("{'description': {", "").replace("}", "")
            masked_inputs.append(masked_text)
            
            # Create label text with the masked values
            label_text = " ".join([str(val) for val in label_data])
            labels.append(label_text)
        
        # Tokenize inputs and labels
        input_encodings = self.tokenizer(masked_inputs, padding=True, truncation=True, return_tensors="pt")
        label_encodings = self.tokenizer(labels, padding=True, truncation=True, return_tensors="pt")
        
        # Set pad tokens in labels to -100 (ignored in loss calculation)
        label_ids = label_encodings['input_ids']
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': label_ids
        }
    
