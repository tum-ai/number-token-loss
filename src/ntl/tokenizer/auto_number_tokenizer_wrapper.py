import re
from typing import List, Any

from transformers import PreTrainedTokenizer


class NumberTokenizerWrapper:
    """
    Wrapper for any tokenizer that adds number token functionality.
    Only tokens that represent actual numbers (integers or decimals) are considered number tokens.
    Special values like 'Inf', 'NaN' etc. are excluded.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, tokenize_on_digit_level: bool = False):
        self.tokenizer = tokenizer
        
        # Add a flag to control digit-level tokenization
        self.tokenize_on_digit_level = tokenize_on_digit_level

        # Store the original methods we'll need to override
        self.original_tokenize = self.tokenizer.tokenize
        
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
        if self.tokenize_on_digit_level:
            return token in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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
            return self.tokenizer.__getattr__(name)
        except AttributeError:
            return getattr(self.tokenizer, name) 
        
    def __setattr__(self, name: str, value: Any) -> None:
        """Fallback to wrapped tokenizer attributes for anything not in wrapper"""
        if name in ['tokenizer', 'num_tokens', 'num_token_ids']:
            super().__setattr__(name, value)
        else:
            setattr(self.tokenizer, name, value)
        
        
    def __len__(self):
        return len(self.tokenizer)
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize text, splitting digit-containing tokens into individual characters."""
        # Use the original tokenize method first
        out = self.original_tokenize(text, **kwargs)
        
        # If digit-level tokenization is disabled, return the original output
        if not self.tokenize_on_digit_level:
            return out
        
        # Process tokens to split digit-containing ones
        out_list = []
        for token in out:
            if bool(re.search(r'\d', token)) and token not in self.additional_special_tokens:
                out_list = out_list + list(token)
            else:
                out_list.append(token)
            
        return out_list

    def __call__(self, *args, **kwargs):
        """Override the __call__ method to intercept tokenization at the highest level."""
        # Save original methods
        original_tokenize_method = self.tokenizer.tokenize 
        
        # Replace with our methods temporarily
        self.tokenizer.tokenize = self.tokenize
        
        try:
            # Process with our custom methods
            result = self.tokenizer(*args, **kwargs)
            return result
        finally:
            # Restore original methods
            self.tokenizer.tokenize = original_tokenize_method

    def encode(self, *args, **kwargs):
        """Override encode to ensure our tokenization is used."""
        # Same pattern as __call__
        original_tokenize_method = self.tokenizer.tokenize
        self.tokenizer.tokenize = self.tokenize
        
        try:
            return self.tokenizer.encode(*args, **kwargs)
        finally:
            self.tokenizer.tokenize = original_tokenize_method

    def encode_plus(self, *args, **kwargs):
        """Override encode_plus to ensure our tokenization is used."""
        original_tokenize_method = self.tokenizer.tokenize
        self.tokenizer.tokenize = self.tokenize
        
        try:
            return self.tokenizer.encode_plus(*args, **kwargs)
        finally:
            self.tokenizer.tokenize = original_tokenize_method

    def batch_encode_plus(self, *args, **kwargs):
        """Override batch_encode_plus to ensure our tokenization is used."""
        original_tokenize_method = self.tokenizer.tokenize
        self.tokenizer.tokenize = self.tokenize
        
        try:
            return self.tokenizer.batch_encode_plus(*args, **kwargs)
        finally:
            self.tokenizer.tokenize = original_tokenize_method

    def __getitem__(self, item):
        return self.tokenizer[item]
    
    def __contains__(self, item):
        return item in self.tokenizer
