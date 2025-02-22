from typing import List, Union, Optional
from transformers import PreTrainedTokenizer
import re


class NumberTokenizerWrapper:
    """
    Wrapper for any tokenizer that adds number token functionality.
    Only tokens that represent actual numbers (integers or decimals) are considered number tokens.
    Special values like 'Inf', 'NaN' etc. are excluded.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
        # Get the vocabulary
        vocab = self.tokenizer.get_vocab()
        
        # Initialize number tokens by finding all tokens that can be converted to floats
        self.num_tokens = []
        self.num_token_ids = []
        
        # Try to convert each token to a float after stripping the space prefix
        for token, id in vocab.items():
            try:
                if self.is_valid_number_token(token):
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

    def is_valid_number_token(self, token: str) -> bool:
        """
        Check if a token represents a valid number (integer or decimal).
        Excludes special values like Inf, NaN, etc.
        """
        # Strip any space prefix
        clean_token = token.lstrip('▁')
        
        # Regular expression for valid number format:
        # - Optional minus sign
        # - One or more digits
        # - Optional decimal point followed by digits
        number_pattern = r'^-?\d+\.?\d*$'
        
        if not re.match(number_pattern, clean_token):
            return False
            
        try:
            value = float(clean_token)
            # Check for special values
            if value in [float('inf'), float('-inf')] or value != value:  # value != value checks for NaN
                return False
            return True
        except ValueError:
            return False

    def decode_number_token(self, token: str, ignore_order: bool = True) -> float:
        """Convert a token to its numerical value."""
        if not self.is_valid_number_token(token):
            raise ValueError(f"Token {token} is not a valid number token")
        return float(token.lstrip('▁'))

    def __getattr__(self, name: str):
        """Fallback to wrapped tokenizer attributes for anything not in wrapper"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.tokenizer, name) 
        
    def __len__(self):
        return len(self.tokenizer)
    
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    def __getitem__(self, item):
        return self.tokenizer[item]
    
    def __contains__(self, item):
        return item in self.tokenizer
