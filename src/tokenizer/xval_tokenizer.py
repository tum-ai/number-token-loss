import re
import numpy as np


class XvalTokenizer:
    def __init__(self, pretrained_tokenizer, num_token="[NUM]"):
        self.tokenizer = pretrained_tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [num_token],
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        })
        self.num_token = num_token
        self.num_token_id = self.tokenizer.convert_tokens_to_ids(num_token)
        print(f"Number token ID: {self.num_token_id}")

    def __call__(self, text, return_attention_mask=False, return_token_type_ids=True):
        if isinstance(text, dict):
            text = text["text"]

        nonum_text, numbers = extract(text, num_token=self.num_token)
        out = self.tokenizer(
            nonum_text, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids
        )
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
