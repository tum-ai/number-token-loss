import copy
import functools
import os
import re
from typing import List, Union
import logging

import numpy as np
import torch

from src.encoding_decoding.numerical_encodings import encoding_to_number
from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer, NUMBER_REGEX

NON_NUMERIC_TOKEN = 10000


class RtTokenizer(NumberEncodingTokenizer):
    def __init__(self, embedding_dim=256, **kwargs):
        super().__init__(**kwargs)

        with open(os.path.join(os.path.dirname(__file__), "..", "..", "regression_transformer_number_tokens.txt"), 'r') as file:
            lines = file.readlines()

        num_tokens = [line.strip() for line in lines]

        self.add_tokens(num_tokens)
        self.num_tokens = num_tokens
        self.num_token_ids = [self.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        self.embedding_dim = embedding_dim

    def get_num_token_ids(self):
        return self.num_token_ids

    def get_num_tokens(self):
        return self.num_tokens

    def decode_number_token(self, token):
        if len(token) == 1 or not (
                token.startswith("_") and token.endswith("_") and token.count("_") == 3
        ):
            raise ValueError(f"no valid rt encoding {token}")

        digit = int(token[1])
        return digit

    def tokenize(self, text: str, add_special_tokens=False, **kwargs) -> List[str]:
        nonum_text, number_tokens = extract(text)
        out = super().tokenize(
            nonum_text, **kwargs
        )
        return out

    def decode_into_human_readable(self, ids: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"]) -> List[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()

        parsed_tokens = np.array([self.convert_ids_to_tokens(ids[i]) for i in range(len(ids))])
        converted_numbers, count_invalid_number_prediction, count_no_number_prediction = self._convert_tokens_to_num_rt(parsed_tokens)

        converted_numbers = [list(filter(lambda x: x not in self.all_special_tokens, decoded_id)) for decoded_id in converted_numbers]
        try:
            decoded = [self.convert_tokens_to_string(tokens) if len(tokens) else "" for tokens in converted_numbers]
        except Exception as e:
            logging.error(f"Error converting tokens to string: {e} for tokens {converted_numbers}")
            decoded = ["" for _ in range(len(converted_numbers))]

        return decoded, count_invalid_number_prediction, count_no_number_prediction

    def _convert_token_to_check_validity(self, token: str) -> int:
        """
        Validates a token and extracts its numeric value if valid. Uses number encoding to allow usage with numpy

        Args:
            token (str): The token to be validated and converted.

        Returns:
            int: The extracted digit index if the token is numeric, 10000 for valid NON_NUMERIC_TOKEN.
        """

        if token.startswith("_") and token.endswith("_"):
            parts = token.strip("_").split("_")
            return int(parts[1])
        else:
            return NON_NUMERIC_TOKEN

    def _convert_tokens_to_num_rt(self, token_array: np.ndarray):
        """
        Converts an array of tokens into numeric values, checking for validity and applying transformations.

        Args:
            token_array (np.ndarray): Array of tokens to be converted.

        Returns:
            tuple: A tuple containing:
                - A lists of RT array with the numeric values or NaNs for invalid sequences.
                - A boolean mask indicating which rows contain valid sequences.
        """
        digit_position_array = np.vectorize(self._convert_token_to_check_validity)(token_array)
        #invalid_count = np.sum(digit_position_array == MALFORMED_RT_TOKEN)

        number_token_array = copy.deepcopy(token_array)
        number_token_array = np.vectorize(functools.partial(encoding_to_number, invalid_strict=False), otypes=[float])(number_token_array)

        result = []
        count_invalid_number_prediction = 0
        count_no_number_prediction = 0
        for row in range(token_array.shape[0]):
            result.append([])
            is_negative = False
            current_number = 0
            for idx in range(token_array.shape[1]):
                # If number token
                if not np.isnan(number_token_array[row][idx]):
                    current_number += number_token_array[row][idx]
                    # If the next token is no number or a number with bigger or equal digit position,
                    # we have to add the current number to the result
                    
                    if idx + 1 == token_array.shape[1] \
                            or np.isnan(number_token_array[row][idx + 1]) \
                            or digit_position_array[row][idx + 1] >= digit_position_array[row][idx]:
                        
                        #Model predicts e.g. (_1_2_, _2_1_) which implicitely denotes zeros / is not a well formed number.
                        #We parse with on a best efforts base
                        if digit_position_array[row][idx]>0:
                            count_invalid_number_prediction+=1
                        
                        result[row].extend(super().tokenize(str(current_number * (-1 if is_negative else 1))))
                        current_number = 0
                        is_negative = False

                        # if next token is indeed also a number, we have to add a whitespace
                        if idx + 1 < token_array.shape[1] and not np.isnan(number_token_array[row][idx + 1]):
                            result[row].append("▁")

                # no number token
                else:
                    token = token_array[row][idx]
                    is_negative = token == "[NEG]"
                    if is_negative:
                        continue
                    result[row].append(token)
        count_no_number_prediction = np.sum(np.all(np.isnan(number_token_array), axis=1))
        return result, count_invalid_number_prediction, count_no_number_prediction


def extract(text):
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

    nonum_text = re.sub(NUMBER_REGEX, replace, text)
    return nonum_text, numbers


if __name__ == "__main__":
    tokenizer = RtTokenizer.from_pretrained("t5-small")
    print(tokenizer._convert_tokens_to_num_rt(np.array([['_1_2_', '_2_1_', '_5_0_'],['_7_2_', '=', '_1_0_']])))
    print(tokenizer.convert_tokens_to_string(tokenizer.tokenize("(3/2)x=54\n3x=108")))