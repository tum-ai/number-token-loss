import re
import numpy as np
from src.tokenizer.abstract_tokenizer import AbstractTokenizer

class RtTokenizer(AbstractTokenizer):
    def __init__(self, special_tokens, num_tokens, embedding_dim, pretrained_tokenizer=None, vocab_files=None, save_file=None):
        super().__init__(special_tokens=special_tokens, num_tokens=num_tokens, embedding_dim=embedding_dim, pretrained_tokenizer=pretrained_tokenizer, vocab_files=vocab_files, save_file=save_file)
        """self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["[NEG]"],
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        })"""

    def __call__(self, text, return_attention_mask=False, return_token_type_ids=True):
        if isinstance(text, dict):
            text = text["text"]

        nonum_text, number_tokens = self.extract(text)
        out = self.tokenizer(
            nonum_text, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids
        )

        ids = np.array(out["input_ids"])
        locs = np.isin(ids, self.num_token_ids)

        number_embeddings = np.array([generate_ne(token, self.embedding_dim) for token in number_tokens])
        num_embed = np.zeros((len(ids), self.embedding_dim)).astype(np.float32)  # Use float32 instead of float16

        num_embed[locs, :] = number_embeddings

        out["number_embeddings"] = num_embed
        out["len"] = len(out["input_ids"])
        return out


    def extract(self, text):
        pattern = r"\s*[\s]*?(\+|\-)?(\d+)(\.)?(\d+)?\s*"
        numbers = []

        def replace(match):
            number = match.group(0).strip()
            tokens = []
            is_negative = number.startswith('-')
            if is_negative:
                number = number[1:]
                tokens.append('[NEG]') # TODO

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
                tokens.append(f"_{digit}_{-i-1}_")
                number.append(f"_{digit}_{-i-1}_")

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