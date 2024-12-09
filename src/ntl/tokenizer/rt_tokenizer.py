import copy
import functools
import os
import re
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import List, Union, Tuple
import logging
import importlib.resources


import numpy as np
import torch

from ntl.encoding_decoding.numerical_encodings import encoding_to_number
from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer, NUMBER_REGEX


NON_NUMERIC_TOKEN = 10000


class RtTokenizer(NumberEncodingTokenizer):
    def __init__(self, embedding_dim=256, **kwargs):
        super().__init__(**kwargs)

        with importlib.resources.open_text(
            "ntl.data", "regression_transformer_number_tokens.txt"
        ) as file:
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

    def decode_number_token(self, token: str, ignore_order: bool = True) -> float:
        return encoding_to_number(token, ignore_order=ignore_order)

    def tokenize(self, text: str, add_special_tokens=False, **kwargs) -> List[str]:
        nonum_text, number_tokens = self.extract(text)
        out = super().tokenize(
            nonum_text, **kwargs
        )
        return out

    def decode_into_human_readable(
            self,
            ids: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"]
    ) -> Tuple[List[str], int, int]:
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()

        parsed_tokens = np.array([self.convert_ids_to_tokens(ids[i]) for i in range(len(ids))])
        (converted_numbers,
        count_invalid_number_prediction,
        count_no_number_prediction_at_all) = self._convert_tokens_to_num_rt(parsed_tokens)

        converted_numbers = [list(filter(lambda x: x not in self.all_special_tokens, decoded_id)) for decoded_id in
                             converted_numbers]
        try:
            decoded = [self.convert_tokens_to_string(tokens) if len(tokens) else "" for tokens in converted_numbers]
        except Exception as e:
            logging.error(f"Error converting tokens to string: {e} for tokens {converted_numbers}")
            decoded = ["" for _ in range(len(converted_numbers))]

        return decoded, count_invalid_number_prediction, count_no_number_prediction_at_all

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
            if len(parts) < 2:
                return NON_NUMERIC_TOKEN
            return int(parts[1])
        else:
            return NON_NUMERIC_TOKEN

    def _convert_tokens_to_num_rt(self, token_array: np.ndarray) -> Tuple[List[List[str]], int, int]:
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

        number_token_array = copy.deepcopy(token_array)
        number_token_array = np.vectorize(functools.partial(encoding_to_number, invalid_strict=False, return_as_decimal=True), otypes=[Decimal])(
            number_token_array)

        result = []
        count_invalid_number_prediction = 0

        for row in range(token_array.shape[0]):
            result.append([])
            is_negative = False
            current_number = Decimal(0)
            for idx in range(token_array.shape[1]):
                # If number token
                if isinstance(number_token_array[row][idx], Decimal):
                    current_number += number_token_array[row][idx]

                    # If the next token is no number or a number with bigger or equal digit position,
                    # we have to add the current number to the result
                    if idx + 1 == token_array.shape[1] \
                            or not isinstance(number_token_array[row][idx + 1], Decimal) \
                            or digit_position_array[row][idx + 1] >= digit_position_array[row][idx]:

                        # Model predicts e.g. (_1_2_, _2_1_) which implicitely denotes zeros / is not a well formed number.
                        # We parse with on a best efforts base
                        if digit_position_array[row][idx] > 0:
                            count_invalid_number_prediction += 1

                        current_number = current_number * (-1 if is_negative else 1)
                        # Normalize the Decimal to remove any trailing zeros
                        current_number = str("{0:.12f}".format(current_number.normalize()).rstrip('0').rstrip('.'))

                        result[row].extend(super().tokenize(current_number))

                        current_number = Decimal(0)
                        is_negative = False

                        # if next token is indeed also a number, we have to add a whitespace
                        if idx + 1 < token_array.shape[1] and isinstance(number_token_array[row][idx + 1], Decimal):
                            result[row].append("▁")

                    # if the next token is a number with digit position smaller than the current one -1, we also
                    # have to count it as invalid
                    elif idx + 1 < token_array.shape[1] \
                            and isinstance(number_token_array[row][idx + 1], Decimal) \
                            and digit_position_array[row][idx + 1] < digit_position_array[row][idx] - 1:
                        count_invalid_number_prediction += 1

                # no number token
                else:
                    token = token_array[row][idx]
                    is_negative = token == "[NEG]"
                    if is_negative:
                        continue
                    result[row].append(token)

        count_no_number_prediction_at_all = np.sum(np.all(np.isnan(number_token_array.astype(float)), axis=1))
        return result, count_invalid_number_prediction, count_no_number_prediction_at_all

    def extract(self, text):
        numbers = []

        # Build a regex pattern that matches any of the special tokens
        special_tokens = self.all_special_tokens
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        special_tokens_pattern = '|'.join(escaped_special_tokens)
        special_tokens_regex = re.compile(special_tokens_pattern)

        def process_part(part):
            def replace(match):
                number = match.group(0).strip()
                tokens = []

                # Remove plus as previously we treated it like a digit
                number = number.lstrip('+-')

                total_digits = len(number.replace('.', ''))
                required_precision = total_digits + 12  # Add 12 for decimal places
                number = Decimal(number)

                # set local context to required precision to avoid floating point errors during rounding
                with localcontext() as ctx:
                    ctx.prec = required_precision
                    # Round the number to 12 decimal places
                    precision = Decimal('1.000000000000')
                    number = number.quantize(precision, rounding=ROUND_HALF_UP)
                    number = str("{0:.12f}".format(number.normalize()).rstrip('0').rstrip('.'))

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
            return re.sub(NUMBER_REGEX, replace, part)

        # Split the text into tokens, keeping the special tokens
        tokens = special_tokens_regex.split(text)
        special_tokens_in_text = special_tokens_regex.findall(text)

        # Now process the tokens
        result = []
        for i, token in enumerate(tokens):
            # Process the token if it's not empty
            if token:
                processed_token = process_part(token)
                result.append(processed_token)
            # Add the special token if it exists
            if i < len(special_tokens_in_text):
                result.append(special_tokens_in_text[i])

        # Join the result to get the final text
        nonum_text = ''.join(result)

        return nonum_text, numbers


if __name__ == "__main__":
    tokenizer = RtTokenizer.from_pretrained("t5-small")
    print(tokenizer._convert_tokens_to_num_rt(np.array([['_1_2_', '_2_1_', '_5_0_'], ['_7_2_', '=', '_1_0_']])))
    print(tokenizer.convert_tokens_to_string(tokenizer.tokenize("(3/2)x=54\n3x=108")))
