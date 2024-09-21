import unittest
from typing import Dict, List

import torch
import torch.nn.functional as F
import transformers

from src.loss_functions.wasserstein_distance_number_token_loss import WassersteinNumberTokenLoss
from src.tokenizer.rt_tokenizer import RtTokenizer
from src.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
import numpy as np


class TestNumberTokenLoss(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.rt_tokenizer = RtTokenizer.from_pretrained("t5-small")
        self.t5_tokenizer = T5Custom_Tokenizer.from_pretrained("t5-small")

        # Get vocab size
        self.config = transformers.T5Config.from_pretrained("t5-small")
        n_new_tokens = len(self.rt_tokenizer) - len(transformers.AutoTokenizer.from_pretrained("t5-small"))
        self.vocab_size = self.config.vocab_size + n_new_tokens

        self.rt_number_token_loss = WassersteinNumberTokenLoss(self.rt_tokenizer, self.vocab_size, self.device)
        self.t5_number_token_loss = WassersteinNumberTokenLoss(self.t5_tokenizer, self.vocab_size, self.device)


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
                {"_1_0_": 1.0, "_2_0_": 1.0, "_0_0_": 0.5, "_3_0_": 1.5},
                {"_1_0_": 1.5, "_2_0_": 1.0, "_0_0_": 0.5, "_3_0_": 1.5},
                {"_1_0_": 1.0, "_2_0_": 1.5, "_0_0_": 0.5, "_3_0_": 1.5},
            ],
        )

        labels = torch.tensor([[
            self.rt_tokenizer.convert_tokens_to_ids("_1_0_"),
            self.rt_tokenizer.convert_tokens_to_ids("_1_0_"),
            self.rt_tokenizer.convert_tokens_to_ids("a")
        ]])

        manual_numbers = [1, 2, 0, 3]
        softmaxed = F.softmax(torch.tensor([1.0, 1.0, 0.5, 1.5]))
        weighted_sum = sum([manual_numbers[i] * s for i, s in enumerate(softmaxed)])
        target = 1.0
        expected_loss = F.mse_loss(torch.tensor([weighted_sum]), torch.tensor([target]))

        ntl_loss = self.rt_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")

        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_loss_with_negative_logits_rt(self):
        logits = self.create_logits(
            self.rt_tokenizer,
            [
                {"_1_0_": -1.0, "_2_0_": 1.0, "_0_0_": 0.5, "_3_0_": -1.5}
            ]
        )

        labels = torch.tensor([[
            self.rt_tokenizer.convert_tokens_to_ids("_2_0_")
        ]])

        manual_numbers = [1, 2, 0, 3]
        softmaxed = F.softmax(torch.tensor([-1.0, 1.0, 0.5, -1.5]))
        weighted_sum = sum([manual_numbers[i] * s for i, s in enumerate(softmaxed)])
        target = 2.0
        expected_loss = F.mse_loss(torch.tensor([weighted_sum]), torch.tensor([target]))

        ntl_loss = self.rt_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")

        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_loss_with_fraction_rt(self):
        logits = self.create_logits(
            self.rt_tokenizer,
            [
                {"_5_-1_": 2.0, "_3_-1_": 1.0, "_2_-1_": 0.5, "_1_-1_": 1.5}
            ]
        )

        labels = torch.tensor([[
            self.rt_tokenizer.convert_tokens_to_ids("_3_-1_")
        ]])

        manual_numbers = [5, 3, 2, 1]
        softmaxed = F.softmax(torch.tensor([2.0, 1.0, 0.5, 1.5]))
        weighted_sum = sum([manual_numbers[i] * s for i, s in enumerate(softmaxed)])
        target = 3
        expected_loss = F.mse_loss(torch.tensor([weighted_sum]), torch.tensor([target]))

        ntl_loss = self.rt_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")
        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_loss_multiple_simple_rt(self):
        # this is 123 guess
        logits = self.create_logits(
            self.rt_tokenizer,
            [
                {"_1_2_": 6.0, "_2_2_": 5.0, "_3_2_": 3.0},
                {"_1_1_": 5.0, "_2_1_": 7.0, "_3_1_": 3.0},
                {"_1_0_": 5.0, "_2_0_": 3.0, "_3_0_": 7.5}
            ]
        )
        # this is 123 label
        labels = torch.tensor([[
            self.rt_tokenizer.convert_tokens_to_ids("_1_2_"),
            self.rt_tokenizer.convert_tokens_to_ids("_2_1_"),
            self.rt_tokenizer.convert_tokens_to_ids("_3_0_")
        ]])
        manual_numbers1 = [1, 2, 3]
        softmaxed1 = F.softmax(torch.tensor([6.0, 5.0, 3.0]))
        weighted_sum1 = sum([manual_numbers1[i] * s for i, s in enumerate(softmaxed1)])

        manual_numbers2 = [1, 2, 3]
        softmaxed2 = F.softmax(torch.tensor([5.0, 7.0, 3.0]))
        weighted_sum2 = sum([manual_numbers2[i] * s for i, s in enumerate(softmaxed2)])

        manual_numbers3 = [1, 2, 3]
        softmaxed3 = F.softmax(torch.tensor([5.0, 3.0, 7.5]))
        weighted_sum3 = sum([manual_numbers3[i] * s for i, s in enumerate(softmaxed3)])

        target = torch.tensor([1, 2, 3])
        expected_loss = F.mse_loss(torch.tensor([weighted_sum1, weighted_sum2, weighted_sum3]), target)

        ntl_loss = self.rt_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")
        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_loss_multiple_mixed_logits_rt(self):
        logits = self.create_logits(
            self.rt_tokenizer,
            [
                {"_1_1_": 2.0, "_2_1_": -1.0},
                {"_1_0_": 0.5, "_0_0_": 1.5},
                {"_5_-1_": 1.0, "_3_-1_": -2.0},
            ]
        )

        labels = torch.tensor([[
            self.rt_tokenizer.convert_tokens_to_ids("_2_1_"),
            self.rt_tokenizer.convert_tokens_to_ids("_1_0_"),
            self.rt_tokenizer.convert_tokens_to_ids("_3_-1_")
        ]])

        # Manual calculation for each token
        manual_numbers1 = [1, 2]
        softmaxed1 = F.softmax(torch.tensor([2.0, -1.0]))
        weighted_sum1 = sum([manual_numbers1[i] * s for i, s in enumerate(softmaxed1)])

        manual_numbers2 = [1, 0]
        softmaxed2 = F.softmax(torch.tensor([0.5, 1.5]))
        weighted_sum2 = sum([manual_numbers2[i] * s for i, s in enumerate(softmaxed2)])

        manual_numbers3 = [5, 3]
        softmaxed3 = F.softmax(torch.tensor([1.0, -2.0]))
        weighted_sum3 = sum([manual_numbers3[i] * s for i, s in enumerate(softmaxed3)])

        target = torch.tensor([2.0, 1.0, 3.0])
        expected_loss = F.mse_loss(torch.tensor([weighted_sum1, weighted_sum2, weighted_sum3]), target)

        ntl_loss = self.rt_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")
        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

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
            self.assertFalse(torch.isnan(self.rt_number_token_loss.nvocab[id]),
                            f"Non-Number token {token} should be nan")
        # All non-number tokens should have a zero embedding
        for token, id in not_number_tokens_to_id.items():
            self.assertTrue(torch.isnan(self.rt_number_token_loss.nvocab[id]),
                            f"Non-Number token {token} should be nan")

    ###################
    # T5 specific tests
    ###################
    def test_loss_with_positive_logits_t5(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"1": 1.0, "2": 1.0, "0": 0.5, "3": 1.5}
            ]
        )

        labels = torch.tensor([[
            self.t5_tokenizer.convert_tokens_to_ids("1")
        ]])

        manual_numbers = [1, 2, 0, 3]
        softmaxed = F.softmax(torch.tensor([1.0, 1.0, 0.5, 1.5]))
        weighted_sum = sum([manual_numbers[i] * s for i, s in enumerate(softmaxed)])
        target = 1.0
        expected_loss = F.mse_loss(torch.tensor([weighted_sum]), torch.tensor([target]))

        ntl_loss = self.t5_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")

        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_loss_with_negative_logits_t5(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"1": -1.0, "2": 1.0, "0": 0.5, "3": -1.5}
            ]
        )

        labels = torch.tensor([[
            self.t5_tokenizer.convert_tokens_to_ids("2")
        ]])

        manual_numbers = [1, 2, 0, 3]
        softmaxed = F.softmax(torch.tensor([-1.0, 1.0, 0.5, -1.5]))
        weighted_sum = sum([manual_numbers[i] * s for i, s in enumerate(softmaxed)])
        target = 2.0
        expected_loss = F.mse_loss(torch.tensor([weighted_sum]), torch.tensor([target]))

        ntl_loss = self.t5_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")

        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_loss_with_fraction_t5(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"5": 2.0, "3": 1.0, "2": 0.5, "1": 1.5}
            ]
        )

        labels = torch.tensor([[
            self.t5_tokenizer.convert_tokens_to_ids("3")
        ]])

        manual_numbers = [5, 3, 2, 1]
        softmaxed = F.softmax(torch.tensor([2.0, 1.0, 0.5, 1.5]))
        weighted_sum = sum([manual_numbers[i] * s for i, s in enumerate(softmaxed)])
        target = 3
        expected_loss = F.mse_loss(torch.tensor([weighted_sum]), torch.tensor([target]))

        ntl_loss = self.t5_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")
        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_loss_multiple_simple_t5(self):
        # this is 123 guess
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"1": 6.0, "2": 5.0, "3": 3.0},
                {"1": 5.0, "2": 7.0, "3": 3.0},
                {"1": 5.0, "2": 3.0, "3": 7.5}
            ]
        )
        # this is 123 label
        labels = torch.tensor([[
            self.t5_tokenizer.convert_tokens_to_ids("1"),
            self.t5_tokenizer.convert_tokens_to_ids("2"),
            self.t5_tokenizer.convert_tokens_to_ids("3")
        ]])
        manual_numbers1 = [1, 2, 3]
        softmaxed1 = F.softmax(torch.tensor([6.0, 5.0, 3.0]))
        weighted_sum1 = sum([manual_numbers1[i] * s for i, s in enumerate(softmaxed1)])

        manual_numbers2 = [1, 2, 3]
        softmaxed2 = F.softmax(torch.tensor([5.0, 7.0, 3.0]))
        weighted_sum2 = sum([manual_numbers2[i] * s for i, s in enumerate(softmaxed2)])

        manual_numbers3 = [1, 2, 3]
        softmaxed3 = F.softmax(torch.tensor([5.0, 3.0, 7.5]))
        weighted_sum3 = sum([manual_numbers3[i] * s for i, s in enumerate(softmaxed3)])

        target = torch.tensor([1, 2, 3])
        expected_loss = F.mse_loss(torch.tensor([weighted_sum1, weighted_sum2, weighted_sum3]), target)

        ntl_loss = self.t5_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")
        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_loss_multiple_mixed_logits_t5(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"1": 2.0, "2": -1.0},
                {"1": 0.5, "0": 1.5},
                {"5": 1.0, "3": -2.0},
            ]
        )

        labels = torch.tensor([[
            self.t5_tokenizer.convert_tokens_to_ids("2"),
            self.t5_tokenizer.convert_tokens_to_ids("1"),
            self.t5_tokenizer.convert_tokens_to_ids("3")
        ]])

        # Manual calculation for each token
        manual_numbers1 = [1, 2]
        softmaxed1 = F.softmax(torch.tensor([2.0, -1.0]))
        weighted_sum1 = sum([manual_numbers1[i] * s for i, s in enumerate(softmaxed1)])

        manual_numbers2 = [1, 0]
        softmaxed2 = F.softmax(torch.tensor([0.5, 1.5]))
        weighted_sum2 = sum([manual_numbers2[i] * s for i, s in enumerate(softmaxed2)])

        manual_numbers3 = [5, 3]
        softmaxed3 = F.softmax(torch.tensor([1.0, -2.0]))
        weighted_sum3 = sum([manual_numbers3[i] * s for i, s in enumerate(softmaxed3)])

        target = torch.tensor([2.0, 1.0, 3.0])
        expected_loss = F.mse_loss(torch.tensor([weighted_sum1, weighted_sum2, weighted_sum3]), target)

        ntl_loss = self.t5_number_token_loss.forward(logits, labels)

        print(f"Expected Loss: {expected_loss.item()}")
        print(f"NumberTokenLoss: {ntl_loss.item()}")
        self.assertAlmostEqual(expected_loss.item(), ntl_loss.item(), places=5)

    def test_nvocab_indexing_t5(self):
        vocab_to_id = self.t5_tokenizer.get_vocab()
        number_tokens = self.t5_tokenizer.get_num_tokens()
        number_tokens_to_id = {token: id for token, id in vocab_to_id.items() if token in number_tokens}
        not_number_tokens_to_id = {token: id for token, id in vocab_to_id.items() if token not in number_tokens}

        # All number tokens should have a non-zero embedding
        for token, id in number_tokens_to_id.items():
            self.assertFalse(torch.isnan(self.t5_number_token_loss.nvocab[id]),
                            f"Non-Number token {token} should be nan")
        # All non-number tokens should have a zero embedding
        for token, id in not_number_tokens_to_id.items():
            self.assertTrue(torch.isnan(self.t5_number_token_loss.nvocab[id]),
                            f"Non-Number token {token} should be nan")

    ###################
    # General tests
    ###################
    def test_loss_with_empty(self):
        self.assertRaises(ValueError,
                          lambda: self.rt_number_token_loss.forward(torch.tensor([]), torch.tensor([])))
        self.assertRaises(ValueError,
                          lambda: self.t5_number_token_loss.forward(torch.tensor([]), torch.tensor([])))
