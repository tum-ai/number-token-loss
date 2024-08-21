from typing import List, Union, Dict

import torch
from transformers import DataCollatorForLanguageModeling

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number

import pdb

import random
import numpy as np

class XvalCookingRecipeCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mask_ratio=0.15, max_span_length=5):
        super().__init__(tokenizer, mlm=True)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_ratio = mask_ratio
        self.max_span_length = max_span_length
        
    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        # Get all encodings
        examples_ = [example['text'] for example in examples]
        encodings = self.tokenizer(examples_, padding=True, truncation=True, return_tensors="pt")
        
        input_ids = encodings['input_ids']
        input_number_embeddings = encodings["number_embeddings"]
        attention_mask = encodings['attention_mask']
        
        number_token_ids = self.tokenizer.get_num_token_ids()
        special_token_ids = self.tokenizer.all_special_ids
        
        # Generate stratified span masking
        input_ids_masked, labels, number_labels = self.stratified_span_masking(input_ids, input_number_embeddings, number_token_ids, special_token_ids)
        #pdb.set_trace()
        
        return {
            'input_ids': input_ids_masked,
            'input_number_embeddings': input_number_embeddings,
            'attention_mask': attention_mask,
            'labels': labels, 
            'number_labels': number_labels
        }

    def stratified_span_masking(self, input_ids, input_number_embeddings, num_token_ids, special_token_ids):
        labels = input_ids.clone()
        input_ids_masked = input_ids.clone()
        number_labels = torch.zeros_like(input_number_embeddings)

        batch_size, seq_length = input_ids.shape
        for i in range(batch_size):
            seq = input_ids[i]
            label_num_mask = torch.isin(seq, torch.tensor(num_token_ids, dtype=torch.long))
            special_token_mask = torch.isin(seq, torch.tensor(special_token_ids, dtype=torch.long))
            
            # Get indices of numerical and word tokens
            number_indices = torch.where(label_num_mask & ~special_token_mask)[0]
            word_indices = torch.where(~label_num_mask & ~special_token_mask)[0]
            
            self.apply_span_masking(input_ids_masked[i], labels[i], number_labels[i], input_number_embeddings[i], number_indices, label_num_mask)
            self.apply_span_masking(input_ids_masked[i], labels[i], number_labels[i], input_number_embeddings[i], word_indices)
        
        return input_ids_masked, labels, number_labels

    def apply_span_masking(self, input_ids_seq, labels_seq, number_labels_seq, input_number_embeddings_seq, token_indices, label_num_mask=None):
        if len(token_indices) == 0:
            return
        
        n_tokens_to_mask = max(1, int(self.mask_ratio * len(token_indices)))
        span_starts = token_indices[torch.randperm(len(token_indices))[:n_tokens_to_mask]]
        
        for start in span_starts:
            span_length = random.randint(1, self.max_span_length)
            end = min(start + span_length, len(input_ids_seq))
            
            #mask span with a special token
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])
            input_ids_seq[start:end] = mask_token_id
            
            # Adjust labels for span masking
            labels_seq[start + 1:end] = -100
            labels_seq[start] = input_ids_seq[start]
            
            # Handle number labels if numerical tokens
            if label_num_mask is not None:
                for idx in range(start, end):
                    if label_num_mask[idx]:
                        number_labels_seq[idx] = input_number_embeddings_seq[idx] 
    
"""
    def __init__(self, tokenizer):
        super().__init__(tokenizer, mlm=True)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        # Get all encodings
        examples_ = [example['text'] for example in examples]
        encodings = self.tokenizer(examples_, padding=True, truncation=True, return_tensors="pt")
        
        # Mask examples (stratify by numbers/words)
        number_token_ids = self.tokenizer.get_num_token_ids()
        special_token_ids = self.tokenizer.all_special_ids
        masked = get_stratified_masking(encodings['input_ids'], number_token_ids, special_token_ids)
        
        input_ids = encodings['input_ids']
        input_number_embeddings = encodings["number_embeddings"]
        attention_mask = encodings['attention_mask']
        
        # Apply the mask to the input_ids
        input_ids_masked = input_ids.clone()
        input_ids_masked[masked] = self.tokenizer.mask_token_id
        
        labels = input_ids.clone()
        labels[~masked] = -100  # We only compute loss on masked tokens

        number_labels = torch.zeros_like(input_number_embeddings)
        number_labels[masked] = input_number_embeddings[masked]

        return {
            'input_ids': input_ids_masked,
            'input_number_embeddings': input_number_embeddings,
            'attention_mask': attention_mask,
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