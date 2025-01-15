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
            per_device_train_batch_size=8,
            max_steps=50000,
            save_steps=500,
            eval_steps=500,
            save_total_limit=1,
            learning_rate=1e-4,
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
        def generate_text(length):
            words = ["calculate", "compute", "determine", "find", "solve", "evaluate", 
                    "number", "value", "result", "sum", "product", "quotient",
                    "large", "small", "medium", "complex", "simple", "basic"]
            return " ".join(np.random.choice(words, size=length))

        def read_json():
            # Generate 100 samples for training
            if "train" in path.lower():
                for i in range(100):
                    length = np.random.randint(3, 21)  # Random length between 3-20 words
                    question = generate_text(length)
                    answer = str(length * 10) # + np.random.randint(-2, 3))  # Some noise around length*10
                    yield {"question": question, "answer": answer}
            # Generate 20 different samples for validation
            elif "val" in path.lower():
                for i in range(20):
                    length = np.random.randint(21, 31)  # Length between 21-30 words
                    question = generate_text(length)
                    answer = str(length * 10)
                    yield {"question": question, "answer": answer}
            # Generate 20 different samples for testing
            else:
                for i in range(20):
                    length = np.random.randint(31, 41)  # Length between 31-40 words
                    question = generate_text(length)
                    answer = str(length * 10)# + np.random.randint(-2, 3))
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
