import re
import numpy as np
from transformers import T5Tokenizer

    
class XvalTokenizer(T5Tokenizer):
    def __init__(self, embedding_dim=256, **kwargs):
        super().__init__(**kwargs)
        num_tokens = ["[NUM]"]
        self.add_tokens(num_tokens)
        self.num_tokens = num_tokens
        self.num_token_ids = [self.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        self.embedding_dim = embedding_dim

    def tokenize(self, text, **kwargs): #TODO
        nonum_text, numbers = extract(text, num_token=self.num_token)
        out = super().tokenize(nonum_text, **kwargs)
        ids = np.array(out["input_ids"])
        locs = ids == self.num_token_id
        num_embed = np.ones(len(ids)).astype(np.float32)  # Use float32 instead of float16
        num_embed[locs] = numbers
        out["numbers"] = num_embed
        out["len"] = len(ids)
        return out


def extract(text, num_token="[NUM]"):
    import re

    # this regular expression is intended to match numerical values in various forms
    # like integers, floating-point numbers, or scientific notation, while avoiding
    # matching numbers that are part of strings.
    pattern = r"(?<!\')-?\d+(\.\d+)?([eE][-+]?\d+)?(?!\'|\d)"

    numbers = []

    def replace(match):
        numbers.append(match.group())
        return "¬"

    nonum_text = re.sub(pattern, replace, text)
    return compress_matrix(nonum_text).replace("¬", num_token), numbers


def compress_matrix(text):
    text = (
        text.replace("¬, ¬", "¬¬")
        .replace("¬, ¬", "¬¬")
        .replace("¬,¬", "¬¬")
        .replace("¬,¬", "¬¬")
    )
    return text
