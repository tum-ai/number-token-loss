from typing import List

from transformers import T5Tokenizer
import numpy as np
import re
import os


class RtTokenizer(T5Tokenizer):
    def __init__(self, embedding_dim=256, **kwargs):
        super().__init__(**kwargs)

        with open(os.path.join(os.path.dirname(__file__), "..", "..", "regression_transformer_number_tokens.txt"), 'r') as file:
            lines = file.readlines()

        num_tokens = [line.strip() for line in lines]


        self.add_special_tokens({"additional_special_tokens": ["[NEG]"]})
            #"pad_token": "[PAD]",
            #"mask_token": "[MASK]",
        #})
        self.add_tokens(num_tokens)
        self.num_tokens = num_tokens
        self.num_token_ids = [self.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        self.embedding_dim = embedding_dim

    def tokenize(self, text: str, add_special_tokens=False, **kwargs) -> List[str]:
        nonum_text, number_tokens = extract(text)
        out = super().tokenize(
            nonum_text, **kwargs
        )
        return out


def extract(text):
    pattern = r"\s*[\s]*?(\+|\-)?(\d+)(\.)?(\d+)?\s*"
    numbers = []

    def replace(match):
        number = match.group(0).strip()
        tokens = []
        is_negative = number.startswith('-')
        if is_negative:
            number = number[1:]
            tokens.append('[NEG]')

        if "." in number:
            integer_part, dot, fractional_part = number.partition('.')
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


def NEFloat(v, p, j):
    return (-1) ** j * v * 10 ** p / (j + 1)


def generate_ne(token, embedding_dim):
    parts = [part for part in token.split('_') if part]
    digit, place = parts
    v = int(digit)
    p = int(place)
    ne = np.zeros(embedding_dim)
    for j in range(embedding_dim):
        ne[j] = NEFloat(v, p, j)
    return ne
