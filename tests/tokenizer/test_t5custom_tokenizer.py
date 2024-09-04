import unittest

from src.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer


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
        ]
        expected_result = [
            '(3/2)x=54 3x=108',
            'Oranges = 12 * 150 = 1800 Nectarines = 16 * 30 = 480 1800 + 480 = 2280 There are 2280 pieces of fruit in total. #### 2280',
            'include 3 10-minute snack breaks each day', 'divide the minutes by 60. 1920 / 60 = 32 hours #### 32',
            'which is 40/100*$400 = $160 The total price',
            'Negativ number: -15.67'
        ]

        result = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        decoded = self.tokenizer.decode_into_human_readable(result["input_ids"])
        self.assertEqual(decoded, expected_result)


if __name__ == "__main__":
    unittest.main()
