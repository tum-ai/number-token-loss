from typing import List, Union, Dict

import torch
from transformers import DataCollatorForLanguageModeling

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number

import numpy as np

class XvalCookingRecipeCollator(DataCollatorForLanguageModeling):
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