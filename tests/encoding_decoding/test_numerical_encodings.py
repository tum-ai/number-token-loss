import unittest

import torch
import transformers

from src.encoding_decoding.numerical_encodings import FloatEncoding, get_float_encoding
from src.tokenizer.rt_tokenizer import RtTokenizer


class TestRtNumericalEncodings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = transformers.T5Config.from_pretrained("t5-small")
        cls.rt_tokenizer = RtTokenizer.from_pretrained("t5-small")
        n_new_tokens = len(cls.rt_tokenizer) - len(transformers.AutoTokenizer.from_pretrained("t5-small"))
        cls.config.vocab_size = cls.config.vocab_size + n_new_tokens
        cls.config.added_vocab = cls.rt_tokenizer.get_added_vocab()
        cls.V_MAX = 3000000000
        cls.number_embeds = FloatEncoding(num_embeddings=cls.config.vocab_size, embedding_dim=cls.config.d_model,
                                          vocab=cls.config.added_vocab, vmax=cls.V_MAX)

    def test_weight_indexing(self):
        weights = self.number_embeds.embedding.weight
        self.assertEqual(weights.shape, (self.config.vocab_size, self.config.d_model))
        self.assertEqual(weights.requires_grad, False)

        vocab_to_id = self.rt_tokenizer.get_vocab()
        number_tokens = self.rt_tokenizer.get_num_tokens()
        number_tokens_to_id = {token: id for token, id in vocab_to_id.items() if token in number_tokens}
        not_number_tokens_to_id = {token: id for token, id in vocab_to_id.items() if token not in number_tokens}

        # All number tokens should have a non-zero embedding
        for token, id in number_tokens_to_id.items():
            if get_float_encoding(token, self.config.d_model, self.V_MAX).abs().sum() == 0:
                self.assertFalse(torch.any(weights[id, :]), f"Token {token} on index {id} has non-zero embedding")
            else:
                self.assertTrue(torch.any(weights[id, :]), f"Token {token} on index {id} has zero embedding")

        # All non-number tokens should have a zero embedding
        for token, id in not_number_tokens_to_id.items():
            self.assertFalse(torch.any(weights[id, :]), f"Token {token} on index {id} has non-zero embedding")



if __name__ == "__main__":
    unittest.main()
