from typing import List, Union, Dict

import torch
from transformers import DataCollatorForLanguageModeling

from ntl.utils.numerical_operations import signed_log


class XvalEra5MLMCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, log_scale):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.log_scale = log_scale
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token = self.tokenizer.additional_special_tokens[0]
        self.mask_token_id = self.tokenizer.additional_special_tokens_ids[0]

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        masked_inputs = []
        labels = []
        
        for example in examples:
            # Get the data lists
            data_lists = example['data']
            
            # Create input by masking the last element of each list
            label_data = str(example.copy()).replace("{'description': {", "").replace("}", "").replace("'", "")
            
            for idx, data_list in enumerate(data_lists):
                # Replace last value with mask token
                data_list[-1] = self.mask_token                
            
            # Convert numerical data to strings for tokenization
            masked_text = str(example).replace("{'description': {", "").replace("}", "").replace("'", "")
            masked_inputs.append(masked_text)
            
            # Create label text with the masked values
            labels.append(label_data)
        
        # Tokenize inputs and labels
        input_encodings = self.tokenizer(masked_inputs, padding=True, truncation=True, return_tensors="pt")
        label_encodings = self.tokenizer(labels, padding=True, truncation=True, return_tensors="pt")

        x = input_encodings['input_ids']
        x_num = input_encodings["number_embeddings"]
        attention_mask = input_encodings['attention_mask']
        mask = x == self.mask_token_id

        y = label_encodings['input_ids']
        y_num = label_encodings["number_embeddings"]

        y[~mask] = -100

        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask}