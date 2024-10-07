# test_run_language_modeling_unittest.py
import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from datasets import Dataset

from src.args import ModelArguments, TrainingArguments, DatasetArguments
from src.run_language_modeling import run_language_modeling
from src.tokenizer.rt_tokenizer import RtTokenizer


class TestRunLanguageModeling(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.temp_dir)

    def generate_model_args(self, number_encoding, number_token_loss, log_scale_embeddings, model_name_or_path=None):
        return ModelArguments(
            model_name_or_path=model_name_or_path,
            config_name="t5-small",
            cache_dir=None,
            number_encoding=number_encoding,
            number_token_loss=number_token_loss,
            log_scale_embeddings=log_scale_embeddings,
        )

    def generate_dataset_args(self):
        return DatasetArguments(
            dataset_name="gsm8k",  # Using gsm8k for faster tests
        )

    def generate_training_args(self, output_dir, do_only_eval=False):
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            num_train_epochs=10,
            save_steps=10,
            save_total_limit=1,
            eval_strategy="no",
            logging_strategy="no",
            learning_rate=1e-4,
            logging_steps=10,
            report_to="none",
            remove_unused_columns=False,
            batch_eval_metrics=True,
            do_only_eval=do_only_eval,
        )

    def mock_load_json_dataset(self, path):
        def read_json():
            yield {"question": "What is 2 + 2?", "answer": "4"}

        return Dataset.from_generator(read_json)

    def mock_load_txt_dataset(self, path):
        def read_txt():
            yield {"question": "What is 2 + 2?", "answer": "4"}

        return Dataset.from_generator(read_txt)

    @mock.patch('src.run_language_modeling.load_json_dataset')
    @mock.patch('src.run_language_modeling.load_txt_dataset')
    def test_model_training(self, mock_load_txt_dataset_fn, mock_load_json_dataset_fn):
        # Mock the dataset loading functions
        mock_load_json_dataset_fn.side_effect = self.mock_load_json_dataset
        mock_load_txt_dataset_fn.side_effect = self.mock_load_txt_dataset

        number_encodings = ["rt", "xval", "none", "none_regression_head"]
        number_token_losses = [True, False]
        log_scale_embeddings_options = [True, False]
        model_names_or_paths = [None, "google-t5/t5-small"]
        xval_bigger_language_heads = [True, False]

        for number_encoding in number_encodings:
            for number_token_loss in number_token_losses:
                for log_scale_embeddings in log_scale_embeddings_options:
                    for model_name_or_path in model_names_or_paths:
                        for xval_bigger_language_head in xval_bigger_language_heads:

                            # Skip invalid combinations
                            if number_encoding in ["xval", "none_regression_head"] and number_token_loss:
                                continue  # NumberTokenLoss is only applicable when number_encoding is not 'xval'

                            if number_encoding != "xval" and xval_bigger_language_head:
                                continue

                            if number_encoding in ["none", "none_regression_head"] and log_scale_embeddings:
                                continue  # Log scaling is only applicable for 'rt' and 'xval' encodings

                            checkpoint_dir = os.path.join(self.temp_dir, "checkpoint-10")

                            # Prepare arguments
                            model_training_args = self.generate_model_args(
                                number_encoding=number_encoding,
                                number_token_loss=number_token_loss,
                                log_scale_embeddings=log_scale_embeddings,
                                model_name_or_path=model_name_or_path,
                            )

                            training_args = self.generate_training_args(output_dir=self.temp_dir)
                            dataset_args = self.generate_dataset_args()
                            model_eval_args = self.generate_model_args(
                                number_encoding=number_encoding,
                                number_token_loss=number_token_loss,
                                log_scale_embeddings=log_scale_embeddings,
                                model_name_or_path=checkpoint_dir,
                            )
                            eval_args = self.generate_training_args(output_dir=self.temp_dir, do_only_eval=True)

                            # Run training
                            with self.subTest(number_encoding=number_encoding,
                                              number_token_loss=number_token_loss,
                                              log_scale_embeddings=log_scale_embeddings):
                                try:
                                    eval_results_expected, _ = run_language_modeling(
                                        model_args=model_training_args,
                                        training_args=training_args,
                                        dataset_args=dataset_args,
                                    )
                                except Exception as e:
                                    logging.error(f"Training failed with exception: {e}", exc_info=True)
                                    self.fail(f"Training failed with exception: {e}")

                                # Check if checkpoint is saved

                                self.assertTrue(os.path.isdir(checkpoint_dir), "Checkpoint directory was not created.")

                                try:
                                    eval_results_val, eval_results_test = run_language_modeling(
                                        model_args=model_eval_args,
                                        training_args=eval_args,
                                        dataset_args=dataset_args,
                                    )
                                except Exception as e:
                                    logging.error(f"Loading model from checkpoint failed with exception: {e}", exc_info=True)
                                    self.fail(f"Loading model from checkpoint failed with exception: {e}")

                                # Clean up checkpoint dir for next iteration
                                shutil.rmtree(checkpoint_dir)

                                # assert equal or both nan
                                np.testing.assert_allclose(eval_results_expected["eval_MAE"], eval_results_val["eval_MAE"], equal_nan=True, err_msg="Validation MAE results do not match.")
                                self.assertEqual(eval_results_expected["eval_token_perplexity"], eval_results_val["eval_token_perplexity"], "Validation perplexity results do not match.")

                                np.testing.assert_allclose(eval_results_expected["eval_MAE"], eval_results_test["eval_MAE"], equal_nan=True, err_msg="Test MAE results do not match.")
                                self.assertEqual(eval_results_expected["eval_token_perplexity"], eval_results_test["eval_token_perplexity"], "Test perplexity results do not match.")


    @mock.patch('src.run_language_modeling.load_json_dataset')
    def test_log_scale_embeddings(self, mock_load_json_dataset_fn):
        # Mock the dataset loading function
        mock_load_json_dataset_fn.side_effect = self.mock_load_json_dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare arguments for model with log scaling
        model_args = self.generate_model_args(
            number_encoding="rt",
            number_token_loss=False,
            log_scale_embeddings=True,
        )
        dataset_args = self.generate_dataset_args()
        training_args = self.generate_training_args(output_dir=self.temp_dir)

        # Run training
        try:
            eval_results_expected, model = run_language_modeling(
                model_args=model_args,
                training_args=training_args,
                dataset_args=dataset_args,
            )
        except Exception as e:
            self.fail(f"Training with log scale embeddings failed with exception: {e}")

        # Check if checkpoint is saved
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoint-10")
        self.assertTrue(os.path.isdir(checkpoint_dir), "Checkpoint directory was not created.")

        tokenizer = RtTokenizer.from_pretrained(checkpoint_dir)

        # Prepare sample input
        sample_input_text = "Calculate the value of 91200005."
        inputs = tokenizer(sample_input_text, return_tensors='pt').to(device)

        # Get embeddings with log scaling
        model.eval()
        with torch.no_grad():
            embeddings_log_scaled = model.get_input_embeddings()(inputs['input_ids'])

        # Now, create another model without log scaling
        # Prepare arguments for model without log scaling
        model_args_no_log = self.generate_model_args(
            number_encoding="rt",
            number_token_loss=False,
            log_scale_embeddings=False,
        )
        training_args_no_log = self.generate_training_args(output_dir=self.temp_dir)

        # Run training for the model without log scaling
        try:
            eval_results_expected, model_no_log = run_language_modeling(
                model_args=model_args_no_log,
                training_args=training_args_no_log,
                dataset_args=dataset_args,
            )
        except Exception as e:
            self.fail(f"Training without log scale embeddings failed with exception: {e}")

        # Get embeddings without log scaling
        model_no_log.eval()
        with torch.no_grad():
            embeddings_no_log = model_no_log.get_input_embeddings()(inputs['input_ids'])

        # Compare embeddings to check if scaling has been applied
        # Find the indices where the number token is located
        num_token_ids = tokenizer.get_num_token_ids()
        number_token_indices = torch.isin(inputs['input_ids'], torch.tensor(num_token_ids).to(device)).squeeze()

        # Check if number_token_indices has any True values
        self.assertTrue(number_token_indices.any(), "No number tokens found in the input.")

        # Extract embeddings for number tokens
        embeddings_log_scaled_numbers = embeddings_log_scaled.squeeze()[number_token_indices]
        embeddings_no_log_numbers = embeddings_no_log.squeeze()[number_token_indices]

        # Assert that the embeddings are different, indicating scaling has been applied
        self.assertFalse(torch.allclose(embeddings_log_scaled_numbers, embeddings_no_log_numbers),
                         "Embeddings with and without log scaling should differ.")

        # Additionally, check that embeddings with log scaling are scaled down
        # since log(large number) is smaller than the number itself
        # For simplicity, compare norms of embeddings
        norm_log_scaled = torch.norm(embeddings_log_scaled_numbers, p=2, dim=-1)
        norm_no_log = torch.norm(embeddings_no_log_numbers, p=2, dim=-1)

        #self.assertTrue(torch.all(norm_log_scaled < norm_no_log),
        #                "Embeddings with log scaling should have smaller norms than without log scaling.")

        # Clean up checkpoint dirs and args files
        shutil.rmtree(checkpoint_dir)


if __name__ == '__main__':
    unittest.main()
