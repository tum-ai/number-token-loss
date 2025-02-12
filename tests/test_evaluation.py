import unittest
import numpy as np
import torch
import transformers
from transformers import EvalPrediction
from ntl.tokenizer.rt_tokenizer import RtTokenizer
from ntl.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from ntl.tokenizer.xval_tokenizer import XvalTokenizer
from ntl.evaluation import CustomMetrics

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
        predictions = [
            "First test 23.0 and # - 1.0",
            "Is 29.0 - 478.2 = # 34.452 correct?",
            "Test text -34*65= # 80",
            "Test 12-12 = # -1 wrong?",
            "Calculation: calculation calculation"
        ]
        labels = [
            "First test 23.0 and # -4.0",
            "Is 29.0 - 478.2 = # 34.452 correct?",
            "Test text -34*65=# 78",
            "Test 12-12 = # 0 wrong?",
            "Calculation: 12 + 12 = # 24"
        ]

        expected_number_results = [
            (-1.0, -4.0),
            (34.452, 34.452),
            (80, 78),
            (-1, 0),
            (np.nan, np.nan),
            ]

        number_results = self.metrics_xval.parse_number_result(predictions, labels)
        self.assertEqual(number_results, expected_number_results)
        number_results = np.concatenate([number_results])
        (
            mae,
            mse,
            r2,
            number_accuracy,
            count_not_produced_valid_results,
            average_count_not_produced_valid_results,
            median_absolute_error,
            log_mae,
            log_r2,
            pearson,
            spearman
        ) = self.metrics_xval.calculate_metrics(number_results, 5)

        expected_mae = np.mean([abs(-1.0 - -4.0), abs(34.452 - 34.452), abs(80 - 78), abs(-1 - 0)])
        expected_mse = np.mean([(4 - 1)**2, 0, (78 - 80)**2, (0 - 1)**2])

        label_mean = np.mean([-4.0, 34.452, 78, 0])
        ss_res = np.sum([(x - y)**2 for x, y in zip([-1.0, 34.452, 80, -1], [-4.0, 34.452, 78, 0])])
        ss_tot = np.sum([(x - label_mean)**2 for x in [-4.0, 34.452, 78, 0]])
        expected_r2 = 1 - (ss_res / ss_tot)

        expected_median_absolute_error = np.median([abs(-1.0 - -4.0), abs(34.452 - 34.452), abs(80 - 78), abs(-1 - 0)])
        expected_log_mae = np.mean([abs(-np.log10(1.0 + 1) - -np.log10(4.0 + 1)), abs(np.log10(34.452 + 1) - np.log10(34.452 + 1)), abs(np.log10(80 + 1) - np.log10(78 + 1)), abs(-np.log10(1 + 1) - np.log10(0 + 1))])

        log_label_mean = np.mean([-np.log10(4.0 + 1), np.log10(34.452 + 1), np.log10(78 + 1), np.log10(0 + 1)])
        log_ss_res = np.sum([(np.sign(x) * np.log10(np.abs(x) + 1) - np.sign(y) * np.log10(np.abs(y) + 1))**2 for x, y in zip([-1.0, 34.452, 80, -1], [-4.0, 34.452, 78, 0])])
        log_ss_tot = np.sum([(np.sign(x) * np.log10(np.abs(x) + 1) - log_label_mean)**2 for x in [-4.0, 34.452, 78, 0]])
        expected_log_r2 = 1 - (log_ss_res / log_ss_tot)

        expected_number_accuracy = 1/5
        expected_count_not_produced_valid_results = 1
        expected_average_count_not_produced_valid_results = 1/5

        expected_pearson = 0.9989029443838093
        expected_spearman = 0.9486832980505139

        np.testing.assert_almost_equal(mae, expected_mae)
        np.testing.assert_almost_equal(mse, expected_mse)
        np.testing.assert_almost_equal(r2, expected_r2)
        np.testing.assert_almost_equal(number_accuracy, expected_number_accuracy)
        np.testing.assert_almost_equal(count_not_produced_valid_results, expected_count_not_produced_valid_results)
        np.testing.assert_almost_equal(average_count_not_produced_valid_results, expected_average_count_not_produced_valid_results)
        np.testing.assert_almost_equal(median_absolute_error, expected_median_absolute_error)
        np.testing.assert_almost_equal(log_mae, expected_log_mae)
        np.testing.assert_almost_equal(log_r2, expected_log_r2)
        np.testing.assert_almost_equal(pearson, expected_pearson)
        np.testing.assert_almost_equal(spearman, expected_spearman)


if __name__ == "__main__":
    unittest.main()
