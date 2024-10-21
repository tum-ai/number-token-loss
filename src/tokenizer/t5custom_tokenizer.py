import os
import re
from typing import List, Union, Tuple

import numpy as np
import torch

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class T5Custom_Tokenizer(NumberEncodingTokenizer):
    """
    Tokenizer for slightly modified T5 encoding. Tokenizes each number as a separate token so that custom loss can be applied.
    Does not add any more additional tokens, as number tokens are already included in the vocabulary.
    """
    def __init__(self, embedding_dim=256, **kwargs):
        super().__init__(**kwargs)

        num_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        self.num_tokens = num_tokens
        self.num_token_ids = [self.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        self.embedding_dim = embedding_dim


    def get_num_token_ids(self):
        return self.num_token_ids

    def get_num_tokens(self):
        return self.num_tokens

    def decode_number_token(self, token: str, ignore_order: bool = True) -> float:
        return float(token)

    def tokenize(self, text: str, add_special_tokens=False, **kwargs) -> List[str]:
        out = super().tokenize(
            text, **kwargs
        )
        out_list = []
        for token in out:
            if bool(re.search(r'\d', token)) and token not in self.additional_special_tokens:
                out_list = out_list + list(token)
            else:
                out_list.append(token)

        return out_list

    def decode_into_human_readable(
            self,
            ids: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"]
    ) -> Tuple[List[str], int, int]:
        decoded = self.batch_decode(ids, skip_special_tokens=True)
        total_invalid_numbers, count_no_number_prediction_at_all = check_number_predictions(decoded)
        return decoded, total_invalid_numbers, count_no_number_prediction_at_all


def check_number_predictions(decoded_preds: List[str]) -> Tuple[int, int]:
    # Greedily match potential numbers with optional signs, digits, commas, and dots. I assumed that there are no dates in the data
    number_pattern = r'[+-]?[\d,.]*\d'

    total_invalid_numbers = 0
    count_no_number_prediction = 0

    for pred in decoded_preds:
        matches = re.findall(number_pattern, pred)

        if not matches:
            count_no_number_prediction += 1
            continue

        for match in matches:
            try:
                parsed_value = float(match)
            except ValueError:
                print(match)
                total_invalid_numbers += 1

    return total_invalid_numbers, count_no_number_prediction
