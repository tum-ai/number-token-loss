import unittest

import numpy as np
import torch

from ntl.tokenizer.xval_tokenizer import XvalTokenizer


class TestEvaluationMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = XvalTokenizer.from_pretrained("t5-small")

    def test_encoding_decoding(self):
        texts = [
           "(3/2)x=54\n3x=108",
           "Oranges = 12 * 150 = 1800\nNectarines = 16 * 30 = 480\n1800 + 480 = 2280\nThere are 2280 pieces of fruit in total.\n#### 2280",
           "include 3 10-minute snack breaks each day",
           "divide the minutes by 60. 1920 / 60 = 32 hours\n#### 32",
           "which is 40/100*$400 = $160\nThe total price",
           "Negativ number: -15.67",
            "[1, 2, 3, -4, 5], [1, 2]"
        ]
        expected_result = [
           '( [NUM] / [NUM] )x= [NUM] [NUM] x= [NUM]',
           'Oranges = [NUM] * [NUM] = [NUM] Nectarines = [NUM] * [NUM] = [NUM] [NUM] + [NUM] = [NUM] There are [NUM] pieces of fruit in total. #### [NUM]',
           'include [NUM] [NUM] -minute snack breaks each day',
           'divide the minutes by [NUM]. [NUM] / [NUM] = [NUM] hours #### [NUM]',
           'which is [NUM] / [NUM] *$ [NUM] = $ [NUM] The total price',
           'Negativ number: [NUM]',
            "[ [NUM], [NUM], [NUM], [NUM], [NUM] ], [ [NUM], [NUM] ]"
        ]
        expected_number_embeddings = torch.tensor([[1.0000e+00, 3.0000e+00, 1.0000e+00, 1.0000e+00, 2.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 5.4000e+01, 3.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0800e+02, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00],
                                             [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.2000e+01, 1.0000e+00, 1.5000e+02,
                                              1.0000e+00, 1.8000e+03, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.6000e+01, 1.0000e+00, 3.0000e+01, 1.0000e+00, 4.8000e+02,
                                              1.8000e+03, 1.0000e+00, 4.8000e+02, 1.0000e+00, 2.2800e+03, 1.0000e+00,
                                              1.0000e+00, 2.2800e+03, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 2.2800e+03,
                                              1.0000e+00],
                                             [1.0000e+00, 3.0000e+00, 1.0000e+01, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00],
                                             [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 6.0000e+01, 1.0000e+00,
                                              1.0000e+00, 1.9200e+03, 1.0000e+00, 1.0000e+00, 6.0000e+01, 1.0000e+00,
                                              3.2000e+01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 3.2000e+01,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00],
                                             [1.0000e+00, 1.0000e+00, 4.0000e+01, 1.0000e+00, 1.0000e+00, 1.0000e+02,
                                              1.0000e+00, 1.0000e+00, 4.0000e+02, 1.0000e+00, 1.0000e+00, 1.6000e+02,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00],
                                             [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              -1.5670e+01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
                                              1.0000e+00, 1.0000e+00, 1.0000e+00],
                                            [1., 1., 1., 1., 2., 1., 1., 3., 1., 1., -4., 1., 1., 5., 1., 1.,
                                             1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
                                            ])

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded = self.tokenizer.batch_decode(result["input_ids"], skip_special_tokens=True)
        self.assertEqual(decoded, expected_result)
        self.assertTrue(torch.equal(result["number_embeddings"], expected_number_embeddings))

    def test_encoding_decoding_decoding_into_human_readable(self):
        texts = [
            "(3/2)x=54\n3x=108",
            "Oranges = 12 * 150 = 1800\nNectarines = 16 * 30 = 480\n1800 + 480 = 2280\nThere are 2280 pieces of fruit in total.\n#### 2280",
            "include 3 10-minute snack breaks each day",
            "divide the minutes by 60. 1920 / 60 = 32 hours\n#### 32",
            "which is 40/100*$400 = $160\nThe total price",
            "Negativ number: -15.67",
        ]

        expected_result = [
            '( 3.0 / 2.0 )x= 54.0 3.0 x= 108.0',
            'Oranges = 12.0 * 150.0 = 1800.0 Nectarines = 16.0 * 30.0 = 480.0 1800.0 + 480.0 = 2280.0 There are 2280.0 pieces of fruit in total. #### 2280.0',
            'include 3.0 10.0 -minute snack breaks each day',
            'divide the minutes by 60.0 . 1920.0 / 60.0 = 32.0 hours #### 32.0',
            'which is 40.0 / 100.0 *$ 400.0 = $ 160.0 The total price',
            'Negativ number: -15.67'
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(result["input_ids"], result["number_embeddings"])
        self.assertEqual(decoded, expected_result)
        self.assertEqual(count_invalid_number_prediction, 0)
        self.assertEqual(count_no_number_prediction, 0)

    def test_decoding_into_human_readable(self):
        token_array = np.array([
            ['▁First', '▁test', '[NUM]', '▁and', '▁', '-', '[NUM]', '</s>', '<pad>', '<pad>', '<pad>'],
            ['▁I', 's', '[NUM]', '▁', '-', '[NUM]', '▁=', '[NUM]', '▁correct', '?', '</s>'],
            ['▁Test', '▁text', '▁', '-', '[NUM]', '▁*', '[NUM]', '▁=', '[NUM]', '</s>', '<pad>'],
            ['▁Test', '[NUM]', '▁', '-', '[NUM]', '▁=', '[NUM]', '▁wrong', '?', '</s>', '<pad>'],
            ['▁Calcul', 'ation', ':', '[NUM]', '▁+', '[NUM]', '▁=', '[NUM]', '</s>', '<pad>', '<pad>']
        ])
        token_ids = self.tokenizer.convert_tokens_to_ids(token_array.flatten())
        token_ids = torch.tensor(token_ids).reshape(token_array.shape)
        number_embeddings = torch.tensor([
            [1.0, 1.0, 23.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 29.0, 1.0, 1.0, 478.20001220703125, 1.0, 34.45199966430664, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 34.0, 1.0, 65.0, 1.0, 78.0, 1.0, 1.0],
            [1.0, 12.0, 1.0, 1.0, 12.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 12.0, 1.0, 12.0, 1.0, 24.0, 1.0, 1.0, 1.0]
        ])

        expected_result = [
            'First test 23.0 and - 4.0',
            'Is 29.0 - 478.2 = 34.452 correct?',
            'Test text - 34.0 * 65.0 = 78.0',
            'Test 12.0 - 12.0 = 0.0 wrong?',
            'Calculation: 12.0 + 12.0 = 24.0'
        ]
        decoded, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(token_ids, number_embeddings)
        self.assertEqual(decoded, expected_result)
        self.assertEqual(count_invalid_number_prediction, 0)
        self.assertEqual(count_no_number_prediction, 0)

        string_array = [
            "First test 23.0 and -4.0",
            "Is 29.0 - 478.2 = 34.452 correct?",
            "Test text -34*65=78",
            "Test 12-12 = 0 wrong?",
            "Calculation: 12 + 12 = 24",
            "No number",
        ]
        expected_result = [
            'First test 23.0 and -4.0',
            'Is 29.0 - 478.2 = 34.452 correct?',
            'Test text -34.0 * 65.0 = 78.0',
            'Test 12.0 -12.0 = 0.0 wrong?',
            'Calculation: 12.0 + 12.0 = 24.0',
            "No number",
        ]
        tokenized = self.tokenizer(string_array, padding=True, truncation=True, return_tensors="pt")
        result, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(tokenized["input_ids"], tokenized["number_embeddings"])
        self.assertEqual(result, expected_result)
        self.assertEqual(count_invalid_number_prediction, 0)
        self.assertEqual(count_no_number_prediction, 1)

    def test_encoding_decoding_special_tokens(self):
        special_token = self.tokenizer.additional_special_tokens[0]
        sentence_end_token = self.tokenizer.eos_token
        pad_token = self.tokenizer.pad_token
        number_token = self.tokenizer.get_num_tokens()[0]
        texts = [
            f"{special_token} 13.87",
            f"{special_token} some text",
        ]

        expected_result_no_skipping = [
            f'{special_token} {number_token} {sentence_end_token}{pad_token}',
            f'{special_token} some text{sentence_end_token}'
        ]

        expected_result_skip_special_tokens = [
            number_token,
            'some text'
        ]

        expected_result_human_readable = [
            '13.87',
            'some text',
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded_no_skipping = self.tokenizer.batch_decode(result["input_ids"], skip_special_tokens=False)
        decoded_skip_special_tokens = self.tokenizer.batch_decode(result["input_ids"], skip_special_tokens=True)
        result_human_readable, _, _ = self.tokenizer.decode_into_human_readable(result["input_ids"], result["number_embeddings"])

        self.assertEqual(decoded_no_skipping, expected_result_no_skipping)
        self.assertEqual(decoded_skip_special_tokens, expected_result_skip_special_tokens)
        self.assertEqual(result_human_readable, expected_result_human_readable)


if __name__ == "__main__":
    unittest.main()
