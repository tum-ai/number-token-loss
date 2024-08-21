from abc import ABC, abstractmethod

from transformers import T5Tokenizer


class NumberEncodingTokenizer(T5Tokenizer, ABC):
    """Abstract base class for number encoding tokenizers based on T5."""

    @abstractmethod
    def get_num_token_ids(self):
        """Should return the list of numerical token IDs."""
        pass
