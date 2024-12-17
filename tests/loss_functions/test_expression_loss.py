import unittest
from typing import Dict, List

import torch
import torch.nn.functional as F
import transformers

from ntl.loss_functions.expression_loss import ExpressionLoss
from ntl.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
import numpy as np
import pdb


class TestNumberTokenLoss(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.t5_tokenizer = T5Custom_Tokenizer.from_pretrained("t5-small")

        # Get vocab size
        self.config = transformers.T5Config.from_pretrained("t5-small")
        n_new_tokens = len(self.t5_tokenizer) - len(
            transformers.AutoTokenizer.from_pretrained("t5-small")
        )
        self.vocab_size = self.config.vocab_size + n_new_tokens

        self.expression_loss = ExpressionLoss(
            self.t5_tokenizer, self.vocab_size, self.device
        )

    def create_logits(
        self, tokenizer, token_logit_value_dict_list: List[Dict[str, float]]
    ) -> torch.Tensor:
        # logits dim = (batch_size, num_tokens, vocab_size)
        logits = torch.full(
            (1, len(token_logit_value_dict_list), self.vocab_size),
            -np.inf,
            dtype=torch.float32,
        )
        for sentence_idx, token_logit_value_dict in enumerate(
            token_logit_value_dict_list
        ):
            for token, prob in token_logit_value_dict.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                logits[0, sentence_idx, token_id] = prob
        return logits

    def test_differentiable(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"<<": 1},
                {"_": 1},
                {"8": 6.0, "2": 5.0, "9": 3.0},
                {"+": 1},
                {"9": 5.0, "3": 7.0, "6": 3.0},
                {"=": 1},
                {"1": 5.0, "2": 3.0, "3": 7.5},
                {"7": 1, "3": 4, "1": 1},
                {">>": 3},
            ],
        )

        logits.requires_grad = True
        labels = self.t5_tokenizer.convert_tokens_to_ids(
            ["<<", "_", "8", "+", "9", "=", "1", "7", ">>"]
        )

        loss = self.expression_loss.forward(
            logits, torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        )
        loss.backward()

        self.assertIsNotNone(
            logits.grad, "Gradients for logits should now have a value."
        )

    def test_convert_logit_seq_to_number(self):
        token_logit_value_dict_list = [
            {"2": 0.5, "1": 0.5},
            {"1": 0.5, "2": 0.5},
        ]

        logits = self.create_logits(self.t5_tokenizer, token_logit_value_dict_list)
        logits = logits[:, :, self.expression_loss.number_tokens]

        # call convert_logit_seq_to_number from the ExpressionLoss instance
        result = self.expression_loss.convert_logit_seq_to_number(logits)

        expected = 16.5
        self.assertAlmostEqual(result.item(), expected, places=2)

    def test_single_input_numbers(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"<<": 1},
                {"_": 1},
                {"8": 6.0, "2": 5.0, "9": 3.0},
                {"+": 1},
                {"9": 5.0, "3": 7.0, "6": 3.0},
                {"=": 1},
                {"1": 5.0, "2": 3.0, "3": 7.5},
                {"7": 1, "3": 4, "1": 1},
                {">>": 3},
            ],
        )

        labels = self.t5_tokenizer.convert_tokens_to_ids(
            ["<<", "_", "8", "+", "9", "=", "1", "7", ">>"]
        )

        loss = self.expression_loss.forward(
            logits, torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        )

        weights = torch.softmax(
            torch.Tensor([[6, 5, 3], [5, 7, 3], [5, 3, 7.5], [1, 4, 1]]), dim=-1
        )

        numbers = torch.Tensor([[8, 2, 9], [9, 3, 6], [1, 2, 3], [7, 3, 1]])

        weighted_numbers = torch.sum(weights * numbers, dim=-1)
        expected_loss = ((weighted_numbers[0] + weighted_numbers[1]) - 17) ** 2

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=2)

    def test_multiple_input_numbers(self):
        logits = self.create_logits(
            self.t5_tokenizer,
            [
                {"<<": 1},
                {"_": 1},
                {"8": 6.0, "2": 5.0, "9": 3.0},
                {"1": 6.0, "3": 5.0, "4": 3.0},
                {"+": 1},
                {"9": 5.0, "3": 7.0, "6": 3.0},
                {"1": 6.0, "3": 5.0, "4": 3.0},
                {"=": 1},
                {"1": 5.0, "2": 3.0, "3": 7.5},
                {"7": 1, "3": 4, "1": 1},
                {"1": 1, "3": 4, "4": 1},
                {">>": 3},
            ],
        )

        labels = self.t5_tokenizer.convert_tokens_to_ids(
            ["<<", "_", "8", "1", "+", "9", "1", "=", "1", "7", "1", ">>"]
        )

        loss = self.expression_loss.forward(
            logits, torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        )

        weights = torch.softmax(
            torch.Tensor(
                [
                    [6, 5, 3],
                    [6, 5, 3],
                    [5, 7, 3],
                    [6, 5, 3],
                    [5, 3, 7.5],
                    [1, 4, 1],
                    [1, 4, 1],
                ]
            ),
            dim=-1,
        )

        numbers = torch.Tensor(
            [
                [8, 2, 9],
                [1, 3, 4],
                [9, 3, 6],
                [1, 3, 4],
                [1, 2, 3],
                [7, 3, 1],
                [1, 3, 4],
            ]
        )

        weighted_numbers = torch.sum(weights * numbers, dim=-1)
        expected_loss = (
            (
                weighted_numbers[0] * 10
                + weighted_numbers[1]
                + weighted_numbers[2] * 10
                + weighted_numbers[3]
            )
            - 171
        ) ** 2

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=2)


if __name__ == "__main__":
    unittest.main()
