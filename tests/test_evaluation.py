import unittest
import numpy as np
import torch
import transformers
from transformers import EvalPrediction
from src.tokenizer.rt_tokenizer import RtTokenizer
from src.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from src.tokenizer.xval_tokenizer import XvalTokenizer
from src.evaluation import CustomMetrics

class TestEvaluationMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize your tokenizers with different encoding schemes
        cls.tokenizer_none = transformers.AutoTokenizer.from_pretrained("t5-small")
        cls.tokenizer_none_custom = T5Custom_Tokenizer.from_pretrained("t5-small")
        cls.tokenizer_xval = XvalTokenizer.from_pretrained("t5-small")
        cls.tokenizer_rt = RtTokenizer.from_pretrained("t5-small")

        # Create instances of the CustomMetrics class for each encoding type
        cls.metrics_none = CustomMetrics(cls.tokenizer_none, number_encoding="none", output_dir="tests/test_output")
        cls.metrics_none_custom = CustomMetrics(cls.tokenizer_none_custom, number_encoding="none", output_dir="tests/test_output")
        cls.metrics_xval = CustomMetrics(cls.tokenizer_xval, number_encoding="xval", output_dir="tests/test_output")
        cls.metrics_rt = CustomMetrics(cls.tokenizer_rt, number_encoding="rt", output_dir="tests/test_output")

    def test_calculate_result_mse(self):
        labels = [
            "First test 23.0 and -4.0",
            "Is 29.0 - 478.2 = 34.452 correct?",
            "Test text -34*65=78",
            "Test 12-12 = 0 wrong?",
            "Calculation: 12 + 12 = 24"
        ]
        predictions = [
            "First test 23.0 and -1.0",
            "Is 29.0 - 478.2 = 34.452 correct?",
            "Test text -34*65=80",
            "Test 12-12 = -1 wrong?",
            "Calculation: calculation calculation"
        ]

        mse = self.metrics_xval.calculate_result_mse(predictions, labels)
        expected_mse = np.array([(4 - 1)**2, 0, (78 - 80)**2, (0 - 1)**2, np.nan])
        np.testing.assert_almost_equal(mse, expected_mse)


if __name__ == "__main__":
    unittest.main()
