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
            "which is 40/100*$400 = $160\nThe total price"
        ]
        expected_result = [
            ['▁(', '_3_0_', '▁', '/', '_2_0_', '▁', ')', 'x', '=', '_5_1_', '_4_0_', '_3_0_', '▁', 'x', '=', '_1_2_', '_0_1_', '_8_0_'],
            ['▁Orange', 's', '▁=', '_1_1_', '_2_0_', '▁*', '_1_2_', '_5_1_', '_0_0_', '▁=', '_1_3_', '_8_2_', '_0_1_', '_0_0_', '▁Ne', 'c', 'tari', 'nes', '▁=', '_1_1_', '_6_0_', '▁*', '_3_1_', '_0_0_', '▁=', '_4_2_', '_8_1_', '_0_0_', '_1_3_', '_8_2_', '_0_1_', '_0_0_', '▁+', '_4_2_', '_8_1_', '_0_0_', '▁=', '_2_3_', '_2_2_', '_8_1_', '_0_0_', '▁There', '▁are', '_2_3_', '_2_2_', '_8_1_', '_0_0_', '▁pieces', '▁of', '▁fruit', '▁in', '▁total', '.', '▁', '##', '##', '_2_3_', '_2_2_', '_8_1_', '_0_0_'],
            ['▁include', '_3_0_', '_1_1_', '_0_0_', '▁', '-', 'minute', '▁snack', '▁breaks', '▁each', '▁day'],
            ['▁divide', '▁the', '▁minutes', '▁by', '_6_1_', '_0_0_',   '▁', '.', '_1_3_', '_9_2_', '_2_1_', '_0_0_', '▁', '/', '_6_1_', '_0_0_', '▁=', '_3_1_', '_2_0_', '▁hours', '▁', '##', '##', '_3_1_', '_2_0_'],
            ['▁which', '▁is', '_4_1_', '_0_0_', '▁', '/', '_1_2_', '_0_1_', '_0_0_', '▁*', '$', '_4_2_', '_0_1_', '_0_0_', '▁=', '▁$', '_1_2_', '_6_1_', '_0_0_', '▁The', '▁total', '▁price']
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
        ]

        expected_result =[
            '( 3.0 / 2.0 )x= 54.0  3.0 x= 108.0',
            'Oranges = 12.0 * 150.0 = 1800.0 Nectarines = 16.0 * 30.0 = 480.0  1800.0 + 480.0 = 2280.0 There are 2280.0 pieces of fruit in total. #### 2280.0',
            'include 3.0  10.0 -minute snack breaks each day',
            'divide the minutes by 60.0 . 1920.0 / 60.0 = 32.0 hours #### 32.0',
            'which is 40.0 / 100.0 *$ 400.0 = $ 160.0 The total price',
            'Negativ number: - 15.67'
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded, _, _ = self.tokenizer.decode_into_human_readable(result["input_ids"])
        self.assertEqual(decoded, expected_result)



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
            'ab 23.0 - 4.0',
            '10.0  200.0 - 478.2',
            'xy 20.0',
            'x 200.0 - 43.0'
        ]
        result, _, _ = self.tokenizer.decode_into_human_readable(token_ids)
        print(result)
        self.assertEqual(result, expected_result)

        string_array = [
            "First test 23.0 and -4.0",
            "Is 29.0 - 478.2 = 34.452 correct?",
            "Test text -34*65=78",
            "Test 12-12 = 0 wrong?",
            "Calculation: 12 + 12 = 24"
        ]
        expected_result = [
            'First test 23.0 and - 4.0',
            'Is 29.0 - 478.2 = 34.452 correct?',
            'Test text - 34.0 * 65.0 = 78.0',
            'Test 12.0 - 12.0 = 0.0 wrong?',
            'Calculation: 12.0 + 12.0 = 24.0'
        ]
        token_ids = self.tokenizer(string_array, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        result, _, _  = self.tokenizer.decode_into_human_readable(token_ids)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
