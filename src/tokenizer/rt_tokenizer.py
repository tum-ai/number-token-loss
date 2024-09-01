import os
import re
from typing import List

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class RtTokenizer(NumberEncodingTokenizer):
    def __init__(self, embedding_dim=256, **kwargs):
        super().__init__(**kwargs)

        with open(os.path.join(os.path.dirname(__file__), "..", "..", "regression_transformer_number_tokens.txt"), 'r') as file:
            lines = file.readlines()

        num_tokens = [line.strip() for line in lines]

        # TODO mask token should not be needed
        mask_token = "[MASK]"
        self.add_tokens([mask_token])
        self.mask_token = mask_token
        self.mask_token_id = self.convert_tokens_to_ids(mask_token)

        self.add_tokens(num_tokens)
        self.num_tokens = num_tokens
        self.num_token_ids = [self.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        self.embedding_dim = embedding_dim

    def get_num_token_ids(self):
        return self.num_token_ids

    def tokenize(self, text: str, add_special_tokens=False, **kwargs) -> List[str]:
        nonum_text, number_tokens = extract(text)
        out = super().tokenize(
            nonum_text, **kwargs
        )
        return out


def extract(text):
    #r"\s*[\s]*?(\+|\-)?(\d+)(\.)?(\d+)?\s*" with r"(\+|\-)?(\d+)(\.)?(\d+)?" to maintain spaces
    #Why are we not using the same strings as xval class (numbers are numbers after all) or the strings from RT?
    pattern = r"(\d+)(\.)?(\d+)?"   
    numbers = []

    def replace(match):
        number = match.group(0).strip()
        tokens = []
        
        #Remove plus as previously we treated it like a digit
        number = number.lstrip('+-')

        if "." in number:
            integer_part, _, fractional_part = number.partition('.')
        else:
            integer_part = number
            fractional_part = []

        z = len(integer_part) - 1
        for digit in integer_part:
            tokens.append(f"_{digit}_{z}_")
            numbers.append(f"_{digit}_{z}_")
            z -= 1
        for i, digit in enumerate(fractional_part):
            tokens.append(f"_{digit}_{-i - 1}_")
            numbers.append(f"_{digit}_{-i - 1}_")

        return " ".join(tokens)

    nonum_text = re.sub(pattern, replace, text)
    return nonum_text, numbers

