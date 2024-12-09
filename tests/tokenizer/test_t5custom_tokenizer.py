import unittest

import transformers

from ntl.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer


class TestEvaluationMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = T5Custom_Tokenizer.from_pretrained("t5-small")

    def test_tokenize(self):
        texts = [
            "(3/2)x=54\n3x=108",
            "Oranges = 12 * 150 = 1800\nNectarines = 16 * 30 = 480\n1800 + 480 = 2280\nThere are 2280 pieces of fruit in total.\n#### 2280",
            "include 3 10-minute snack breaks each day",
            "divide the minutes by 60. 1920 / 60 = 32 hours\n#### 32",
            "which is 40/100*$400 = $160\nThe total price"
        ]

        expected_result = [
            ['▁(', '3', '/', '2', ')', 'x', '=', '5', '4', '▁', '3', 'x', '=', '1', '0', '8'],
            ['▁Orange', 's', '▁=', '▁', '1', '2', '▁*', '▁', '1', '5', '0', '▁=', '▁', '1', '8', '0', '0', '▁Ne', 'c', 'tari', 'nes', '▁=', '▁', '1', '6', '▁*', '▁', '3', '0', '▁=', '▁', '4', '8', '0', '▁', '1', '8', '0', '0', '▁+', '▁', '4', '8', '0', '▁=', '▁', '2', '2', '8', '0', '▁There', '▁are', '▁', '2', '2', '8', '0', '▁pieces', '▁of', '▁fruit', '▁in', '▁total', '.', '▁', '##', '##', '▁', '2', '2', '8', '0'],
            ['▁include', '▁', '3', '▁', '1', '0', '-', 'minute', '▁snack', '▁breaks', '▁each', '▁day'],
            ['▁divide', '▁the', '▁minutes', '▁by', '▁', '6', '0', '.', '▁', '1', '9', '2', '0', '▁', '/', '▁', '6', '0', '▁=', '▁', '3', '2', '▁hours', '▁', '##', '##', '▁', '3', '2'],
            ['▁which', '▁is', '▁', '4', '0', '/', '1', '0', '0', '*', '$', '4', '0', '0', '▁=', '▁', '$', '1', '6', '0', '▁The', '▁total', '▁price']
        ]

        result = [self.tokenizer.tokenize(text) for text in texts]
        self.assertEqual(result, expected_result)


    def test_encoding_decoding(self):
        texts = [
            "(3/2)x=54\n3x=108",
            "Oranges = 12 * 150 = 1800\nNectarines = 16 * 30 = 480\n1800 + 480 = 2280\nThere are 2280 pieces of fruit in total.\n#### 2280",
            "include 3 10-minute snack breaks each day",
            "divide the minutes by 60. 1920 / 60 = 32 hours\n#### 32",
            "which is 40/100*$400 = $160\nThe total price",
            "Negativ number: -15.67",
        ]

        expected_result = [
            '(3/2)x=54 3x=108',
            'Oranges = 12 * 150 = 1800 Nectarines = 16 * 30 = 480 1800 + 480 = 2280 There are 2280 pieces of fruit in total. #### 2280',
            'include 3 10-minute snack breaks each day',
            'divide the minutes by 60. 1920 / 60 = 32 hours #### 32',
            'which is 40/100*$400 = $160 The total price',
            'Negativ number: -15.67'
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded = self.tokenizer.batch_decode(result["input_ids"], skip_special_tokens=True)
        self.assertEqual(decoded, expected_result)

    def test_encoding_decoding_decoding_into_human_readable(self):
        texts = [
            "(3/2)x=54\n3x=108",
            "Oranges = 12 * 150 = 1800\nNectarines = 16 * 30 = 480\n1800 + 480 = 2280\nThere are 2280 pieces of fruit in total.\n#### 2280",
            "include 3 10-minute snack breaks each day",
            "divide the minutes by 60. 1920 / 60 = 32 hours\n#### 32",
            "which is 40/100*$400 = $160\nThe total price",
            "Negativ number: -15.67",
            "Invalid Number 1.111.1",
            "No Number Prediction at all",
        ]
        expected_result = [
            '(3/2)x=54 3x=108',
            'Oranges = 12 * 150 = 1800 Nectarines = 16 * 30 = 480 1800 + 480 = 2280 There are 2280 pieces of fruit in total. #### 2280',
            'include 3 10-minute snack breaks each day', 'divide the minutes by 60. 1920 / 60 = 32 hours #### 32',
            'which is 40/100*$400 = $160 The total price',
            'Negativ number: -15.67',
            'Invalid Number 1.111.1',
            'No Number Prediction at all'
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(result["input_ids"])
        self.assertEqual(decoded, expected_result)
        self.assertEqual(count_invalid_number_prediction, 1)
        self.assertEqual(count_no_number_prediction, 1)

    def test_encoding_decoding_special_tokens(self):
        special_token = self.tokenizer.additional_special_tokens[0]
        sentence_end_token = self.tokenizer.eos_token
        pad_token = self.tokenizer.pad_token
        texts = [
            f"{special_token} 13.87",
            f"{special_token} some text",
        ]

        expected_result_no_skipping = [
            f"{special_token} 13.87{sentence_end_token}",
            f"{special_token} some text{sentence_end_token}{pad_token}{pad_token}{pad_token}{pad_token}",
        ]
        expected_result_skip_special_tokens = [
            "13.87",
            "some text",
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded_no_skipping = self.tokenizer.batch_decode(result["input_ids"], skip_special_tokens=False)
        decoded_skip_special_tokens = self.tokenizer.batch_decode(result["input_ids"], skip_special_tokens=True)

        self.assertEqual(decoded_no_skipping, expected_result_no_skipping)
        self.assertEqual(decoded_skip_special_tokens, expected_result_skip_special_tokens)

    def test_encoding_decoding_special_tokens_vanilla_tokenizer(self):
        vanilla_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
        special_token = vanilla_tokenizer.additional_special_tokens[0]
        sentence_end_token = vanilla_tokenizer.eos_token
        texts = [
            f"{special_token}13.87",
            f"{special_token}some text",
        ]

        expected_result_no_skipping = [
            f"{special_token} 13.87{sentence_end_token}",
            f"{special_token} some text{sentence_end_token}",
        ]
        expected_result_skip_special_tokens = [
            "13.87",
            "some text",
        ]

        result = vanilla_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded_no_skipping = vanilla_tokenizer.batch_decode(result["input_ids"], skip_special_tokens=False)
        decoded_skip_special_tokens = vanilla_tokenizer.batch_decode(result["input_ids"], skip_special_tokens=True)

        self.assertEqual(decoded_no_skipping, expected_result_no_skipping)
        self.assertEqual(decoded_skip_special_tokens, expected_result_skip_special_tokens)


if __name__ == "__main__":
    unittest.main()
