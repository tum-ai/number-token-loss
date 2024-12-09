import unittest
from time import time
from typing import Dict, List
import os
import pytest

import numpy as np
import torch
import transformers

from src.loss_functions.abs_diff_number_token_loss import AbsDiffNumberTokenLoss
from src.loss_functions.wasserstein_distance_number_token_loss import (
    WassersteinNumberTokenLoss,
)
from src.tokenizer.rt_tokenizer import RtTokenizer
from src.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer

def skiptest():
    """Decorator to skip tests when running in GitHub Actions."""
    return pytest.mark.skipif(
        os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipped on GitHub Actions due to high memory usage"
    )

class TestAbsDiffNumberTokenLoss(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.rt_tokenizer = RtTokenizer.from_pretrained("t5-small")
        self.t5_tokenizer = T5Custom_Tokenizer.from_pretrained("t5-small")

        # Get vocab size
        self.config = transformers.T5Config.from_pretrained("t5-small")
        n_new_tokens = len(self.rt_tokenizer) - len(transformers.AutoTokenizer.from_pretrained("t5-small"))
        self.vocab_size = self.config.vocab_size + n_new_tokens

        self.rt_number_token_loss_wasserstein = WassersteinNumberTokenLoss(
            self.rt_tokenizer, self.vocab_size, self.device, order_numbers=False
        )
        self.t5_number_token_loss_wasserstein = WassersteinNumberTokenLoss(
            self.t5_tokenizer, self.vocab_size, self.device, order_numbers=True
        )
        self.rt_number_token_loss_absdiff = AbsDiffNumberTokenLoss(self.rt_tokenizer, self.vocab_size, self.device)
        self.t5_number_token_loss_absdiff = AbsDiffNumberTokenLoss(self.t5_tokenizer, self.vocab_size, self.device)

    def create_logits(self, tokenizer, token_logit_value_dict_list: List[Dict[str, float]]) -> torch.Tensor:
        # logits dim = (batch_size, num_tokens, vocab_size)
        logits = torch.full((1, len(token_logit_value_dict_list), self.vocab_size), -np.inf, dtype=torch.float32)
        for sentence_idx, token_logit_value_dict in enumerate(token_logit_value_dict_list):
            for token, prob in token_logit_value_dict.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                logits[0, sentence_idx, token_id] = prob
        return logits

    ###################
    # Regression Transformer (RT) specific tests
    ###################
    def test_loss_with_positive_logits_rt(self):
        logits = self.create_logits(
            self.rt_tokenizer,
            [
                {"_1_0_": 1.0, "_2_0_": 1.2, "_0_0_": 0.5, "_3_0_": 1.5},
                {"_1_0_": 1.5, "_2_0_": 1.2, "_0_0_": 0.5, "_3_0_": 1.5},
                {"_1_0_": 1.0, "_2_0_": 1.2, "_0_0_": 0.5, "_3_0_": 1.5},
            ],
        )

        labels = torch.tensor(
            [
                [
                    self.rt_tokenizer.convert_tokens_to_ids("_1_0_"),
                    self.rt_tokenizer.convert_tokens_to_ids("_1_0_"),
                    self.rt_tokenizer.convert_tokens_to_ids("a"),
                ]
            ]
        )

        # compare abs diff loss to wasserstein loss and ensure they are the same
        abs_diff_loss = self.rt_number_token_loss_absdiff.forward(logits, labels)
        wasserstein_loss = self.rt_number_token_loss_wasserstein.forward(logits, labels)

        print(f"Expected Loss: {wasserstein_loss.item()}")
        print(f"Abs diff loss: {abs_diff_loss.item()}")

        self.assertAlmostEqual(abs_diff_loss.item(), wasserstein_loss.item(), places=5)

    def test_loss_multiple_mixed_logits_rt(self):
        logits = self.create_logits(
            self.rt_tokenizer,
            [
                {"_0_1_": -4.0, "_1_1_": 2.0, "_2_1_": -1.0},
                {"_0_0_": 1.5, "_1_0_": 0.5, "_2_0_": 1.2},
                {"_4_-1_": -2.0, "_5_-1_": 1.0, "_6_-1_": -2.5},
            ],
        )

        labels = torch.tensor(
            [
                [
                    self.rt_tokenizer.convert_tokens_to_ids("_2_1_"),
                    self.rt_tokenizer.convert_tokens_to_ids("_1_0_"),
                    self.rt_tokenizer.convert_tokens_to_ids("_4_-1_"),
                ]
            ]
        )

        # compare abs diff loss to wasserstein loss and ensure they are the same
        abs_diff_loss = self.rt_number_token_loss_absdiff.forward(logits, labels)
        wasserstein_loss = self.rt_number_token_loss_wasserstein.forward(logits, labels)

        print(f"Expected Loss: {wasserstein_loss.item()}")
        print(f"Abs diff loss: {abs_diff_loss.item()}")

        self.assertAlmostEqual(abs_diff_loss.item(), wasserstein_loss.item(), places=5)

    def test_loss_comparison_rt(self):
        logits = self.create_logits(
            self.rt_tokenizer,
            [
                {"_0_1_": 1.0, "_1_1_": 2.0, "_2_1_": 1.0, "_3_1_": 1.5},
                {"_0_1_": 1.5, "_1_1_": 1.0, "_2_1_": 1.5, "_3_1_": 1},
                {"_0_1_": 1.0, "_1_1_": 1.0, "_2_1_": 1.5, "_3_1_": 1.5},
                {"_0_1_": 1.0, "_1_1_": 1.0, "_2_1_": 1.0, "_3_1_": 2},
            ],
        )

        labels = torch.tensor(
            [
                [
                    self.rt_tokenizer.convert_tokens_to_ids("_1_1_"),
                    self.rt_tokenizer.convert_tokens_to_ids("_1_1_"),
                    self.rt_tokenizer.convert_tokens_to_ids("_1_1_"),
                    self.rt_tokenizer.convert_tokens_to_ids("_1_1_"),
                ]
            ]
        )

        wasserstein_loss_1 = self.rt_number_token_loss_absdiff.forward(logits[:, 0:1], labels[:, 0:1])
        wasserstein_loss_2 = self.rt_number_token_loss_absdiff.forward(logits[:, 1:2], labels[:, 1:2])
        wasserstein_loss_3 = self.rt_number_token_loss_absdiff.forward(logits[:, 2:3], labels[:, 2:3])
        wasserstein_loss_4 = self.rt_number_token_loss_absdiff.forward(logits[:, 3:4], labels[:, 3:4])

        self.assertLess(wasserstein_loss_1.item(), wasserstein_loss_2.item())
        self.assertLess(wasserstein_loss_2.item(), wasserstein_loss_3.item())
        self.assertLess(wasserstein_loss_3.item(), wasserstein_loss_4.item())

    def test_nvocab_indexing_rt(self):
        # test if in nvocab nans or numbers
        # instead of weights, nvocab
        # insteead of checking for -0 and suim just check for

        vocab_to_id = self.rt_tokenizer.get_vocab()
        number_tokens = self.rt_tokenizer.get_num_tokens()
        number_tokens_to_id = {token: id for token, id in vocab_to_id.items() if token in number_tokens}
        not_number_tokens_to_id = {token: id for token, id in vocab_to_id.items() if token not in number_tokens}

        # All number tokens should have a non-zero embedding
        for token, id in number_tokens_to_id.items():
            self.assertFalse(
                torch.isnan(self.rt_number_token_loss_absdiff.nvocab[id]), f"Non-Number token {token} should be nan"
            )
        # All non-number tokens should have a zero embedding
        for token, id in not_number_tokens_to_id.items():
            self.assertTrue(
                torch.isnan(self.rt_number_token_loss_absdiff.nvocab[id]), f"Non-Number token {token} should be nan"
            )

    ###################
    # T5 specific tests
    ###################
    def test_loss_with_positive_logits_t5(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"1": 1.0, "2": 1.2, "0": 0.5, "3": 1.5},
                {"1": 1.5, "2": 1.2, "0": 0.5, "3": 1.5},
                {"1": 1.0, "2": 1.2, "0": 0.5, "3": 1.5},
            ],
        )

        labels = torch.tensor(
            [
                [
                    self.t5_tokenizer.convert_tokens_to_ids("1"),
                    self.t5_tokenizer.convert_tokens_to_ids("1"),
                    self.t5_tokenizer.convert_tokens_to_ids("a"),
                ]
            ]
        )

        # compare abs diff loss to wasserstein loss and ensure they are the same
        abs_diff_loss = self.t5_number_token_loss_absdiff.forward(logits, labels)
        wasserstein_loss = self.t5_number_token_loss_wasserstein.forward(logits, labels)

        print(f"Expected Loss: {wasserstein_loss.item()}")
        print(f"Abs diff loss: {abs_diff_loss.item()}")

        self.assertAlmostEqual(abs_diff_loss.item(), wasserstein_loss.item(), places=5)

    def test_loss_multiple_mixed_logits_t5(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"0": -4.0, "1": 2.0, "2": -1.0},
                {"0": 1.5, "1": 0.5, "2": 1.2},
                {"3": -2.0, "4": 1.0, "5": -2.5},
            ],
        )

        labels = torch.tensor(
            [
                [
                    self.t5_tokenizer.convert_tokens_to_ids("2"),
                    self.t5_tokenizer.convert_tokens_to_ids("1"),
                    self.t5_tokenizer.convert_tokens_to_ids("3"),
                ]
            ]
        )

        # compare abs diff loss to wasserstein loss and ensure they are the same
        abs_diff_loss = self.t5_number_token_loss_absdiff.forward(logits, labels)
        wasserstein_loss = self.t5_number_token_loss_wasserstein.forward(logits, labels)

        print(f"Expected Loss: {wasserstein_loss.item()}")
        print(f"Abs diff loss: {abs_diff_loss.item()}")

        self.assertAlmostEqual(abs_diff_loss.item(), wasserstein_loss.item(), places=3)

    def test_loss_comparison_t5(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"0": 1.0, "1": 2.0, "2": 1.0, "3": 1.5},
                {"0": 1.5, "1": 1.0, "2": 1.5, "3": 1},
                {"0": 1.0, "1": 1.0, "2": 1.5, "3": 1.5},
                {"0": 1.0, "1": 1.0, "2": 1.0, "3": 2},
            ],
        )

        labels = torch.tensor(
            [
                [
                    self.t5_tokenizer.convert_tokens_to_ids("1"),
                    self.t5_tokenizer.convert_tokens_to_ids("1"),
                    self.t5_tokenizer.convert_tokens_to_ids("1"),
                    self.t5_tokenizer.convert_tokens_to_ids("1"),
                ]
            ]
        )

        wasserstein_loss_1 = self.t5_number_token_loss_absdiff.forward(logits[:, 0:1], labels[:, 0:1])
        wasserstein_loss_2 = self.t5_number_token_loss_absdiff.forward(logits[:, 1:2], labels[:, 1:2])
        wasserstein_loss_3 = self.t5_number_token_loss_absdiff.forward(logits[:, 2:3], labels[:, 2:3])
        wasserstein_loss_4 = self.t5_number_token_loss_absdiff.forward(logits[:, 3:4], labels[:, 3:4])

        self.assertLess(wasserstein_loss_1.item(), wasserstein_loss_2.item())
        self.assertLess(wasserstein_loss_2.item(), wasserstein_loss_3.item())
        self.assertLess(wasserstein_loss_3.item(), wasserstein_loss_4.item())

    def test_nvocab_indexing_t5(self):
        vocab_to_id = self.t5_tokenizer.get_vocab()
        number_tokens = self.t5_tokenizer.get_num_tokens()
        number_tokens_to_id = {token: id for token, id in vocab_to_id.items() if token in number_tokens}
        not_number_tokens_to_id = {token: id for token, id in vocab_to_id.items() if token not in number_tokens}

        # All number tokens should have a non-zero embedding
        for token, id in number_tokens_to_id.items():
            self.assertFalse(
                torch.isnan(self.t5_number_token_loss_absdiff.nvocab[id]), f"Non-Number token {token} should be nan"
            )
        # All non-number tokens should have a zero embedding
        for token, id in not_number_tokens_to_id.items():
            self.assertTrue(
                torch.isnan(self.t5_number_token_loss_absdiff.nvocab[id]), f"Non-Number token {token} should be nan"
            )

    ###################
    # General tests
    ###################
    def test_loss_with_empty(self):
        self.assertRaises(
            ValueError, lambda: self.rt_number_token_loss_absdiff.forward(torch.tensor([]), torch.tensor([]))
        )
        self.assertRaises(
            ValueError, lambda: self.t5_number_token_loss_absdiff.forward(torch.tensor([]), torch.tensor([]))
        )

    @skiptest()
    def test_runtime_speed(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"0": -4.0, "1": 2.0, "2": -1.0},
                {"0": 1.5, "1": 0.5, "2": 1.2},
                {"3": -2.0, "4": 1.0, "5": -2.5},
            ],
        )

        labels = torch.tensor(
            [
                [
                    self.t5_tokenizer.convert_tokens_to_ids("2"),
                    self.t5_tokenizer.convert_tokens_to_ids("1"),
                    self.t5_tokenizer.convert_tokens_to_ids("3"),
                ]
            ]
        )
        batch_size: int = 128
        logits = logits.repeat(batch_size, 100, 1)
        labels = labels.repeat(batch_size, 100)
        
        runs: int = 10
        start = time()
        for _ in range(runs):
            self.t5_number_token_loss_absdiff.forward(logits, labels)
        a_time = (time() - start) / runs
        
        start = time()
        for _ in range(runs):
            self.t5_number_token_loss_wasserstein.forward(logits, labels)
        w_time = (time() - start) / runs
        shape_str = " x ".join(list(map(str, list(logits.shape))))
        print(f"\n\n****Speed Average for BS x T x V = {shape_str}***\n\tWasserstein: {w_time:.4f}\n\tAbsdiff: {a_time:.4f}")