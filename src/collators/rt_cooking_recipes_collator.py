from typing import List, Union, Dict

import torch
from transformers import DataCollatorForLanguageModeling

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number

import numpy as np
class RtCookingRecipeCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, mlm=True)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        # Extract text from each example
        examples_ = [example['text'] for example in examples]
        # Tokenize the examples with padding and truncation
        encodings = self.tokenizer(examples_, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        input_ids = encodings['input_ids']
        # Get token IDs for numerical tokens and special tokens
        number_token_ids = self.tokenizer.get_num_token_ids()
        special_token_ids = self.tokenizer.all_special_ids
        # Generate a mask (stratify by numbers and words) 
        masked = get_stratified_masking(input_ids, number_token_ids, special_token_ids)
        
        # clone input_ids and apply mask token
        input_ids_masked = input_ids.clone()
        input_ids_masked[masked] = self.tokenizer.mask_token_id
        
        # Create labels, only masked tokens loss computation
        labels = input_ids.clone()
        labels[~masked] = -100 

        # Create number labels for the numerical tokens
        label_num_mask = torch.isin(input_ids, torch.tensor(number_token_ids, dtype=torch.long))
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[label_num_mask])
        number_values = torch.tensor([encoding_to_number(token) for token in tokens], dtype=torch.float)
        number_labels = torch.zeros_like(input_ids, dtype=torch.float)
        number_labels[label_num_mask] = number_values

        # TODO: print input_ids_masked in the debugger --> they are all masked, and I have not yet figured out why
        import pdb; pdb.set_trace()
        return {
            'input_ids': input_ids_masked,
            'attention_mask': encodings['attention_mask'],
            'labels': labels,
            'number_labels': number_labels
        }

def get_stratified_masking(input_ids, num_token_ids, special_token_ids):
    # create mask to identify numerical and special tokens
    label_num_mask = torch.isin(input_ids, torch.tensor(num_token_ids, dtype=torch.long))
    special_token_mask = torch.isin(input_ids, torch.tensor(special_token_ids, dtype=torch.long))
    
    # indices of numerical and word tokens
    number_indices = torch.where(label_num_mask & ~special_token_mask)[0]
    word_indices = torch.where(~label_num_mask & ~special_token_mask)[0]

    # randomly select 15% of numerical and word tokens to mask
    masked_numbers = rnd_select(number_indices, 0.15)
    masked_words = rnd_select(word_indices, 0.15)
    
    masked_indices = torch.cat([masked_numbers, masked_words], dim=0)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[masked_indices] = True
    return mask

def rnd_select(indices, prob):
    count = int(prob * len(indices))
    if count == 0:
        return torch.tensor([], dtype=torch.long)
    return indices[torch.randperm(len(indices))[:count]]