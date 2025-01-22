import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch
from datasets import Dataset
import numpy as np

from ntl.args import ModelArguments, TrainingArguments, DatasetArguments
from ntl.run_language_modeling import run_language_modeling


class TestNumberHead(unittest.TestCase):
    def setUp(self):
        # Configure logging to show all levels
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Create a temporary directory for the test
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.temp_dir)
        # Reset logging to default
        logging.getLogger().setLevel(logging.WARNING)

    def generate_model_args(self):
        return ModelArguments(
            model_name_or_path="google-t5/t5-small",
            config_name="t5-small",
            cache_dir=None,
            number_encoding="none_regression_head",
            log_scale_embeddings=False,
        )

    def generate_dataset_args(self):
        return DatasetArguments(
            dataset_name="gsm8k",
        )

    def generate_training_args(self, output_dir):
        use_cpu = not torch.cuda.is_available()
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=32,
            max_steps=1000,
            save_steps=200,
            eval_steps=200,
            save_total_limit=1,
            learning_rate=5e-4,
            logging_steps=50,
            eval_strategy="steps",
            report_to="none",
            remove_unused_columns=False,
            batch_eval_metrics=True,
            do_only_eval=False,
            language_modelling="mlm",
            eval_on_start=True,
            use_cpu=use_cpu,
        )

    def mock_load_json_dataset(self, path):
        def generate_simple_math_question(a, b):
            return f"What is {a} plus {b}?", str(a + b)

        def read_json():
            numbers = range(1, 31)
            train_numbers = np.random.choice(numbers, 20, replace=False)
            val_numbers = [n for n in numbers if n not in train_numbers]
            if "train" in path.lower():
                # Generate 100 simple addition problems with numbers 1-20
                for _ in range(100):
                    a = np.random.choice(train_numbers)
                    b = np.random.choice(train_numbers)
                    question, answer = generate_simple_math_question(a, b)
                    yield {"question": question, "answer": answer}
            else:
                # Generate 20 validation/test problems with similar range
                for _ in range(20):
                    a = np.random.choice(val_numbers)
                    b = np.random.choice(val_numbers)
                    question, answer = generate_simple_math_question(a, b)
                    yield {"question": question, "answer": answer}

        return Dataset.from_generator(read_json)

    def mock_load_txt_dataset(self, path):
        # We can reuse the same logic for txt datasets
        return self.mock_load_json_dataset(path)

    @mock.patch('ntl.run_language_modeling.load_json_dataset')
    @mock.patch('ntl.run_language_modeling.load_txt_dataset')
    def test_model_training(self, mock_load_txt_dataset_fn, mock_load_json_dataset_fn):
        # Mock the dataset loading functions
        mock_load_json_dataset_fn.side_effect = self.mock_load_json_dataset
        mock_load_txt_dataset_fn.side_effect = self.mock_load_txt_dataset

        checkpoint_dir = os.path.join(self.temp_dir, "checkpoint-1000")

        # Prepare arguments
        model_training_args = self.generate_model_args()

        training_args = self.generate_training_args(output_dir=self.temp_dir)
        dataset_args = self.generate_dataset_args()

        try:
            eval_results, _ = run_language_modeling(
                model_args=model_training_args,
                training_args=training_args,
                dataset_args=dataset_args,
            )
        except Exception as e:
            logging.error(f"Training failed with exception: {e}", exc_info=True)
            self.fail(f"Training failed with exception: {e}")

        # Check if checkpoint is saved

        self.assertTrue(os.path.isdir(checkpoint_dir), "Checkpoint directory was not created.")


if __name__ == '__main__':
    unittest.main()
