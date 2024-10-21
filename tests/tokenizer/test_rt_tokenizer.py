import unittest

import numpy as np
import torch

from src.tokenizer.rt_tokenizer import RtTokenizer


class TestEvaluationMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = RtTokenizer.from_pretrained("t5-small")

    def test_tokenize(self):
        texts = [
            "(3/2)x=54\n3x=108",
            "Oranges = 12 * 150 = 1800\nNectarines = 16 * 30 = 480\n1800 + 480 = 2280\nThere are 2280 pieces of fruit in total.\n#### 2280",
            "include 3 10-minute snack breaks each day",
            "divide the minutes by 60. 1920 / 60 = 32 hours\n#### 32",
            "which is 40/100*$400 = $160\nThe total price",
            "13.8703074063372",
            "13.8703074063376",
        ]
        expected_result = [
            ['▁(', '_3_0_', '▁', '/', '_2_0_', '▁', ')', 'x', '=', '_5_1_', '_4_0_', '_3_0_', '▁', 'x', '=', '_1_2_', '_0_1_', '_8_0_'],
            ['▁Orange', 's', '▁=', '_1_1_', '_2_0_', '▁*', '_1_2_', '_5_1_', '_0_0_', '▁=', '_1_3_', '_8_2_', '_0_1_', '_0_0_', '▁Ne', 'c', 'tari', 'nes', '▁=', '_1_1_', '_6_0_', '▁*', '_3_1_', '_0_0_', '▁=', '_4_2_', '_8_1_', '_0_0_', '_1_3_', '_8_2_', '_0_1_', '_0_0_', '▁+', '_4_2_', '_8_1_', '_0_0_', '▁=', '_2_3_', '_2_2_', '_8_1_', '_0_0_', '▁There', '▁are', '_2_3_', '_2_2_', '_8_1_', '_0_0_', '▁pieces', '▁of', '▁fruit', '▁in', '▁total', '.', '▁', '##', '##', '_2_3_', '_2_2_', '_8_1_', '_0_0_'],
            ['▁include', '_3_0_', '_1_1_', '_0_0_', '▁', '-', 'minute', '▁snack', '▁breaks', '▁each', '▁day'],
            ['▁divide', '▁the', '▁minutes', '▁by', '_6_1_', '_0_0_',   '▁', '.', '_1_3_', '_9_2_', '_2_1_', '_0_0_', '▁', '/', '_6_1_', '_0_0_', '▁=', '_3_1_', '_2_0_', '▁hours', '▁', '##', '##', '_3_1_', '_2_0_'],
            ['▁which', '▁is', '_4_1_', '_0_0_', '▁', '/', '_1_2_', '_0_1_', '_0_0_', '▁*', '$', '_4_2_', '_0_1_', '_0_0_', '▁=', '▁$', '_1_2_', '_6_1_', '_0_0_', '▁The', '▁total', '▁price'],
            ['_1_1_', '_3_0_', '_8_-1_', '_7_-2_', '_0_-3_', '_3_-4_', '_0_-5_', '_7_-6_', '_4_-7_', '_0_-8_', '_6_-9_', '_3_-10_', '_3_-11_', '_7_-12_'],
            ['_1_1_', '_3_0_', '_8_-1_', '_7_-2_', '_0_-3_', '_3_-4_', '_0_-5_', '_7_-6_', '_4_-7_', '_0_-8_', '_6_-9_', '_3_-10_', '_3_-11_', '_8_-12_']
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
            '( _3_0_ / _2_0_ )x= _5_1_ _4_0_ _3_0_ x= _1_2_ _0_1_ _8_0_',
            'Oranges = _1_1_ _2_0_ * _1_2_ _5_1_ _0_0_ = _1_3_ _8_2_ _0_1_ _0_0_ Nectarines = _1_1_ _6_0_ * _3_1_ _0_0_ = _4_2_ _8_1_ _0_0_ _1_3_ _8_2_ _0_1_ _0_0_ + _4_2_ _8_1_ _0_0_ = _2_3_ _2_2_ _8_1_ _0_0_ There are _2_3_ _2_2_ _8_1_ _0_0_ pieces of fruit in total. #### _2_3_ _2_2_ _8_1_ _0_0_',
            'include _3_0_ _1_1_ _0_0_ -minute snack breaks each day',
            'divide the minutes by _6_1_ _0_0_. _1_3_ _9_2_ _2_1_ _0_0_ / _6_1_ _0_0_ = _3_1_ _2_0_ hours #### _3_1_ _2_0_',
            'which is _4_1_ _0_0_ / _1_2_ _0_1_ _0_0_ *$ _4_2_ _0_1_ _0_0_ = $ _1_2_ _6_1_ _0_0_ The total price',
            'Negativ number: - _1_1_ _5_0_ _6_-1_ _7_-2_',
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
            "0.009",
            "12",
            "12.01",
            "0.0221",
            "-0.000042",
            "10035.2",
            "13987028330851034.9999881",
        ]

        expected_result =[
            '( 3 / 2 )x= 54  3 x= 108',
            'Oranges = 12 * 150 = 1800 Nectarines = 16 * 30 = 480  1800 + 480 = 2280 There are 2280 pieces of fruit in total. #### 2280',
            'include 3  10 -minute snack breaks each day',
            'divide the minutes by 60 . 1920 / 60 = 32 hours #### 32',
            'which is 40 / 100 *$ 400 = $ 160 The total price',
            'Negativ number: - 15.67',
            "0.009",
            "12",
            "12.01",
            "0.0221",
            "- 0.000042",
            "10035.2",
            "13987028330851034.9999881",
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(result["input_ids"])
        self.assertEqual(decoded, expected_result)
        self.assertEqual(count_invalid_number_prediction, 0)
        self.assertEqual(count_no_number_prediction, 0)



    def test_decoding_into_human_readable(self):
        token_array = np.array([
            #            23 -> valid               -4.0 -> valid
            ['▁ab', '_2_1_', '_3_0_', '▁', '-', '_4_0_', '</s>', '<pad>', '<pad>'],
            #  not valid               -478.2 -> valid
            ['_1_1_', '_2_2_', '▁', '-', '_4_2_', '_7_1_', '_8_0_', '_2_-1_', '</s>'],
            #        20 -> valid               not valid, but will now be converted to </unk> and does therefore not count as invalid
            ['▁', 'x', 'y', '_2_1_', '_0_0_', '▁', '_3_','</s>', '<pad>'],
            #       not valid                    -43.0 -> valid
            ['▁', 'x', '_2_2_', '_0_0_', '▁', '-', '_4_1_', '_3_0_', '</s>']

        ])
        token_ids = self.tokenizer.convert_tokens_to_ids(token_array.flatten())
        token_ids = torch.tensor(token_ids).reshape(token_array.shape)

        expected_result = [
            'ab 23 - 4',
            '10  200 - 478.2',
            'xy 20',
            'x 200 - 43'
        ]
        result, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(token_ids)
        self.assertEqual(result, expected_result)
        self.assertEqual(count_invalid_number_prediction, 3)
        self.assertEqual(count_no_number_prediction, 0)

        string_array = [
            "First test 23.0 and -4.0",
            "Is 29.0 - 478.2 = 34.452 correct?",
            "Test text -34*65=78",
            "Test 12-12 = 0 wrong?",
            "Calculation: 12 + 12 = 24",
            "No Number",
        ]
        expected_result = [
            'First test 23 and - 4',
            'Is 29 - 478.2 = 34.452 correct?',
            'Test text - 34 * 65 = 78',
            'Test 12 - 12 = 0 wrong?',
            'Calculation: 12 + 12 = 24',
            "No Number",
        ]
        token_ids = self.tokenizer(string_array, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        result, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(token_ids)
        self.assertEqual(result, expected_result)
        self.assertEqual(count_invalid_number_prediction, 0)
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
            f'{special_token} _1_1_ _3_0_ _8_-1_ _7_-2_ {sentence_end_token}',
            f'{special_token} some text{sentence_end_token}{pad_token}{pad_token}'
        ]

        expected_result_skip_special_tokens = [
            '_1_1_ _3_0_ _8_-1_ _7_-2_',
            'some text'
        ]

        expected_result_human_readable = [
            '13.87',
            'some text',
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded_no_skipping = self.tokenizer.batch_decode(result["input_ids"], skip_special_tokens=False)
        decoded_skip_special_tokens = self.tokenizer.batch_decode(result["input_ids"], skip_special_tokens=True)
        result_human_readable, _, _ = self.tokenizer.decode_into_human_readable(result["input_ids"])

        self.assertEqual(decoded_no_skipping, expected_result_no_skipping)
        self.assertEqual(decoded_skip_special_tokens, expected_result_skip_special_tokens)
        self.assertEqual(result_human_readable, expected_result_human_readable)


if __name__ == "__main__":
    unittest.main()
