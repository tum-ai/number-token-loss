import unittest
import torch
from src.number_token_loss import NumberTokenLoss
from src.tokenizer.rt_tokenizer import RtTokenizer

class TestNumberTokenLoss(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.tokenizer = RtTokenizer.from_pretrained("t5-small")
        self.number_token_loss = NumberTokenLoss(self.tokenizer, self.device)
        self.vocab_size = len(self.tokenizer.get_vocab())

    def create_logits(self, probs):
        logits = torch.full((1, len(probs), self.vocab_size), -1e9)
        for i, prob in enumerate(probs):
            token_id = self.tokenizer.convert_tokens_to_ids(f"_{prob[0]}_{prob[1]}_")
            logits[0, i, token_id] = prob[2]
        return logits

    def test_forward_basic(self):
        logits = self.create_logits([(5, 0, 10.0)])
        labels = torch.tensor([[self.tokenizer.convert_tokens_to_ids("_5_0_")]])
        loss = self.number_token_loss.forward(logits, labels)
        expected_loss = 0.0  # Perfect prediction
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_forward_edge_case_empty_logits(self):
        # Test with empty logits
        logits = torch.empty(0, 0, len(self.tokenizer.get_vocab()))
        labels = torch.tensor([[]])
        with self.assertRaises(Exception):
            self.number_token_loss.forward(logits, labels)

    def test_forward_edge_case_empty_labels(self):
        # Test with empty labels
        logits = torch.randn(1, 0, len(self.tokenizer.get_vocab()))
        labels = torch.tensor([[]])
        with self.assertRaises(Exception):
            self.number_token_loss.forward(logits, labels)

    def test_forward_non_digit_label(self):
        # Test with a non-digit label
        logits = torch.randn(1, 1, len(self.tokenizer.get_vocab()))
        non_digit_token = self.tokenizer.convert_tokens_to_ids("[MASK]")
        labels = torch.tensor([[non_digit_token]])
        loss = self.number_token_loss.forward(logits, labels)
        self.assertTrue(torch.isnan(torch.tensor(loss)))

    def test_forward_multiple_tokens(self):
        logits = self.create_logits([(1, 0, 10.0), (2, 0, 10.0), (3, 0, 10.0)])
        labels = torch.tensor([[
            self.tokenizer.convert_tokens_to_ids("_1_0_"),
            self.tokenizer.convert_tokens_to_ids("_2_0_"),
            self.tokenizer.convert_tokens_to_ids("_3_0_")
        ]])
        loss = self.number_token_loss.forward(logits, labels)
        expected_loss = 0.0  # Perfect predictions
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_forward_batch_processing(self):
        logits = torch.cat([
            self.create_logits([(1, 0, 10.0), (2, 0, 10.0), (3, 0, 10.0)]),
            self.create_logits([(2, 0, 10.0), (1, 0, 10.0), (2, 0, 10.0)])
        ])
        labels = torch.tensor([
            [
                self.tokenizer.convert_tokens_to_ids("_1_0_"),
                self.tokenizer.convert_tokens_to_ids("_2_0_"),
                self.tokenizer.convert_tokens_to_ids("_3_0_")
            ],
            [
                self.tokenizer.convert_tokens_to_ids("_2_0_"),
                self.tokenizer.convert_tokens_to_ids("_1_0_"),
                self.tokenizer.convert_tokens_to_ids("_2_0_")
            ]
        ])
        loss = self.number_token_loss.forward(logits, labels)
        expected_loss = 0.0  # Perfect predictions
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_negative_numbers(self):
        logits = self.create_logits([(5, 0, 10.0), (5, 0, 10.0)])
        labels = torch.tensor([[
            self.tokenizer.convert_tokens_to_ids("[NEG]"),
            self.tokenizer.convert_tokens_to_ids("_5_0_")
        ]])
        loss = self.number_token_loss.forward(logits, labels)
        expected_loss = 0.0
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_fractional_numbers(self):
        logits = self.create_logits([(3, 0, 10.0), (1, -1, 10.0), (4, -2, 10.0)])
        labels = torch.tensor([[
            self.tokenizer.convert_tokens_to_ids("_3_0_"),
            self.tokenizer.convert_tokens_to_ids("_1_-1_"),
            self.tokenizer.convert_tokens_to_ids("_4_-2_")
        ]])
        loss = self.number_token_loss.forward(logits, labels)
        expected_loss = 0.0
        self.assertAlmostEqual(loss, expected_loss, places=5)