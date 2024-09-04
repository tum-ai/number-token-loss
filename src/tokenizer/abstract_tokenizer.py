from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch
from transformers import T5Tokenizer

# NUMBER_REGEX = r"(\-)?(\d+)(\.)?(\d+)?"
NUMBER_REGEX = r"(\d+)(\.\d+)?"


class NumberEncodingTokenizer(T5Tokenizer, ABC):
    """Abstract base class for number encoding tokenizers based on T5."""

    @abstractmethod
    def get_num_token_ids(self) -> List[int]:
        """Should return the list of numerical token IDs."""
        pass

    @abstractmethod
    def get_num_tokens(self) -> List[str]:
        pass

    @abstractmethod
    def decode_number_token(self, token: str) -> float:
        pass

    @abstractmethod
    def decode_into_human_readable(self, ids: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"]) -> List[str]:
        pass
