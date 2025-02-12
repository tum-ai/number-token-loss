from transformers import AutoTokenizer
from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
import re
from typing import List, Union, Tuple
import numpy as np
import torch


class AutoNumberTokenizer(NumberEncodingTokenizer):
    """
    Wrapper for AutoTokenizer that adds number token functionality.
    Any token that can be converted to a float (after stripping the '▁' prefix) 
    is considered a number token.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get the vocabulary
        vocab = self.get_vocab()
        
        # Initialize number tokens by finding all tokens that can be converted to floats
        self.num_tokens = []
        self.num_token_ids = []
        
        # Try to convert each token to a float after stripping the space prefix
        for token, id in vocab.items():
            try:
                self.decode_number_token(token)
                self.num_tokens.append(token)
                self.num_token_ids.append(id)
            except ValueError:
                continue
                
        if not self.num_tokens:
            raise ValueError("No number tokens found in vocabulary")

    def get_num_token_ids(self) -> List[int]:
        return self.num_token_ids

    def get_num_tokens(self) -> List[str]:
        return self.num_tokens

    def decode_number_token(self, token: str, ignore_order: bool = True) -> float:
        # Strip any space prefix before converting to float
        clean_token = token.lstrip('▁')
        try:
            return float(clean_token)
        except ValueError:
            raise ValueError(f"Cannot convert token {token} to float")

    def decode_into_human_readable(
            self,
            ids: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"]
    ) -> Tuple[List[str], int, int]:
        decoded = self.batch_decode(ids, skip_special_tokens=True)
        total_invalid_numbers = 0
        count_no_number_prediction = 0
        
        number_pattern = r'[+-]?[\d,.]*\d'
        
        for pred in decoded:
            matches = re.findall(number_pattern, pred)
            
            if not matches:
                count_no_number_prediction += 1
                continue
                
            for match in matches:
                try:
                    float(match)
                except ValueError:
                    total_invalid_numbers += 1
                    
        return decoded, total_invalid_numbers, count_no_number_prediction

 