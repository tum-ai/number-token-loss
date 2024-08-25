import unittest
import torch
from src.number_token_loss import NumberTokenLoss
from src.tokenizer.rt_tokenizer import RtTokenizer


class TestNumberTokenLoss(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward_basic(self):
        pass

    def test_forward_with_order(self):
        pass

    def test_forward_edge_case_empty_logits(self):
        pass

    def test_forward_edge_case_empty_labels(self):
        pass

    def test_forward_non_digit_label(self):
        pass
