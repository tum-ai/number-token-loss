import os
import re
from typing import List

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class T5Custom_Tokenizer(NumberEncodingTokenizer):
    """
    Tokenizer for slightly modified T5 encoding. Tokenizes each number as a separate token so that custom loss can be applied.
    Does not add any more additional tokens, as number tokens are already included in the vocabulary.
    """
    def __init__(self, embedding_dim=256, **kwargs):
        super().__init__(**kwargs)

        num_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # TODO remove NEG token?
        self.add_special_tokens({"additional_special_tokens": ["[NEG]"]})

        # TODO mask token should not be needed
        mask_token = "[MASK]"
        self.add_tokens([mask_token])
        self.mask_token = mask_token
        self.mask_token_id = self.convert_tokens_to_ids(mask_token)

        self.num_tokens = num_tokens
        self.num_token_ids = [self.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        self.embedding_dim = embedding_dim


    def get_num_token_ids(self):
        return self.num_token_ids

    def get_num_tokens(self):
        return self.num_tokens

    def decode_number_token(self, token):
        return float(token)

    def tokenize(self, text: str, add_special_tokens=False, **kwargs) -> List[str]:
        out = super().tokenize(
            text, **kwargs
        )
        out_list = []
        for token in out:
            if bool(re.search(r'\d', token)):
                out_list = out_list + list(token)
            else:
                out_list.append(token)

        return out_list    