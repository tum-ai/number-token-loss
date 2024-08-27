from abc import ABC, abstractmethod

from transformers import T5Tokenizer
from typing import List

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
