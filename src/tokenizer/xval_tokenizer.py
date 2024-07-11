import re
import numpy as np
from src.tokenizer.abstract_tokenizer import AbstractTokenizer

class XvalTokenizer(AbstractTokenizer):
    def __init__(self, special_tokens, num_tokens, embedding_dim, pretrained_tokenizer=None, vocab_files=None, save_file=None):
        super().__init__(special_tokens=special_tokens, num_tokens=num_tokens, embedding_dim=embedding_dim, pretrained_tokenizer=pretrained_tokenizer, vocab_files=vocab_files, save_file=save_file)
        
    def __call__(self, text, return_attention_mask=False, return_token_type_ids=True):
        if isinstance(text, dict):
            text = text["text"]

        nonum_text, numbers = self.extract(text)#, num_tokens=self.num_tokens[0]) # TODO
        out = self.tokenizer(
            nonum_text, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids
        )
        ids = np.array(out["input_ids"])
        locs = ids == self.num_token_ids[0] # TODO
        num_embed = np.ones(len(ids)).astype(np.float32)  # Use float32 instead of float16
        num_embed[locs] = numbers
        out["numbers"] = num_embed
        out["len"] = len(ids)
        return out


    def extract(self, text):#, num_token="[NUM]"):
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
        return compress_matrix(nonum_text).replace("¬", self.num_tokens[0]), numbers


def compress_matrix(text):
    text = (
        text.replace("¬, ¬", "¬¬")
        .replace("¬, ¬", "¬¬")
        .replace("¬,¬", "¬¬")
        .replace("¬,¬", "¬¬")
    )
    return text
