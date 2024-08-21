from typing import List, Union, Dict

import pdb 

import torch
from transformers import DataCollatorForLanguageModeling

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number

import random
import numpy as np

class RtCookingRecipeCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mask_ratio=0.15, max_span_length=5):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_ratio = mask_ratio
        self.max_span_length = max_span_length

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        # Get text from each example
        examples_ = [example['text'] for example in examples]
        encodings = self.tokenizer(examples_, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        input_ids = encodings['input_ids']
        number_token_ids = self.tokenizer.get_num_token_ids()
        special_token_ids = self.tokenizer.all_special_ids
        
        # Generate stratified span masking
        input_ids_masked, labels = self.stratified_span_masking(input_ids, number_token_ids, special_token_ids)
        #pdb.set_trace()
        # Create number labels for the numerical tokens
        label_num_mask = torch.isin(input_ids, torch.tensor(number_token_ids, dtype=torch.long))
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[label_num_mask])
        number_values = torch.tensor([encoding_to_number(token) for token in tokens], dtype=torch.float)
        number_labels = torch.zeros_like(input_ids, dtype=torch.float)
        number_labels[label_num_mask] = number_values
        
        return {
            'input_ids': input_ids_masked,
            'attention_mask': encodings['attention_mask'],
            'labels': labels,
            'number_labels': number_labels
        }

    def stratified_span_masking(self, input_ids, num_token_ids, special_token_ids):
        labels = input_ids.clone()
        input_ids_masked = input_ids.clone()

        batch_size, _ = input_ids.shape
        for i in range(batch_size):
            seq = input_ids[i]
            label_num_mask = torch.isin(seq, torch.tensor(num_token_ids, dtype=torch.long))
            special_token_mask = torch.isin(seq, torch.tensor(special_token_ids, dtype=torch.long))
            
            number_indices = torch.where(label_num_mask & ~special_token_mask)[0]
            word_indices = torch.where(~label_num_mask & ~special_token_mask)[0]
            
            # apply stratified span masking 
            self.apply_span_masking(input_ids_masked[i], labels[i], number_indices)
            self.apply_span_masking(input_ids_masked[i], labels[i], word_indices)
        
        return input_ids_masked, labels

    def apply_span_masking(self, input_ids_seq, labels_seq, token_indices):
        if len(token_indices) == 0:
            return
        
        n_tokens_to_mask = max(1, int(self.mask_ratio * len(token_indices)))
        span_starts = token_indices[torch.randperm(len(token_indices))[:n_tokens_to_mask]]
        
        for start in span_starts:
            span_length = random.randint(1, self.max_span_length)
            end = min(start + span_length, len(input_ids_seq))
            
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])
            input_ids_seq[start:end] = mask_token_id
            
            # Adjust labels for span masking
            labels_seq[start + 1:end] = -100
            labels_seq[start] = input_ids_seq[start]
            
"""
    
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
        #pdb.set_trace()
        # Generate a msask (stratify by numbers and words) 
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

        #pdb.set_trace()
        return {
            'input_ids': input_ids_masked,
            'attention_mask': encodings['attention_mask'],
            'labels': labels,
            'number_labels': number_labels
        }


def get_stratified_masking(input_ids, num_token_ids, special_token_ids):
    batch_masks = []
    
    for input_ids_seq in input_ids:
        #identify numerical and special tokens
        label_num_mask = torch.isin(input_ids_seq, torch.tensor(num_token_ids, dtype=torch.long))
        special_token_mask = torch.isin(input_ids_seq, torch.tensor(special_token_ids, dtype=torch.long))
        
        # indices of numerical and word tokens
        number_indices = torch.where(label_num_mask & ~special_token_mask)[0]
        word_indices = torch.where(~label_num_mask & ~special_token_mask)[0]

        #randomly select 15% of numerical and word tokens to mask
        masked_numbers = rnd_select(number_indices, 0.15)
        masked_words = rnd_select(word_indices, 0.15)
        
        masked_indices = torch.cat([masked_numbers, masked_words], dim=0)
        mask = torch.zeros_like(input_ids_seq, dtype=torch.bool)
        mask[masked_indices] = True
        
        batch_masks.append(mask)
    
    #pdb.set_trace()
    return torch.stack(batch_masks)

def rnd_select(indices, prob):
    count = int(prob * len(indices))
    if count == 0:
        return torch.tensor([], dtype=torch.long)
    return indices[torch.randperm(len(indices))[:count]]
    
"""