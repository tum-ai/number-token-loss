import unittest
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from ntl.utils.label_smoother import GaussianLabelSmoother


class TestGaussianLabelSmoother(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_length = 3
        self.num_classes = 5
        self.ignore_index = -100

        # Create random logits
        self.logits = torch.randn(self.batch_size, self.seq_length, self.num_classes, requires_grad=True)

        # Create labels with some ignore_index
        self.labels = torch.tensor([
            [1, 2, self.ignore_index],
            [0, self.ignore_index, 4]
        ])

        # Set device for testing 
        self.device = torch.device('cpu')  

    def tearDown(self):
        # Reset gradients after each test
        if self.logits.grad is not None:
            self.logits.grad.zero_()
            
    def test_sigma_zero_equals_cross_entropy(self):
        """
        Test that GaussianLabelSmoother with sigma=0 behaves identically to standard cross entropy loss.
        """
        smoother = GaussianLabelSmoother(sigma=0.0, ignore_index=self.ignore_index, selector=None)

        # Compute loss with GaussianLabelSmoother
        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # Compute standard cross entropy loss ignoring the ignore_index
        # Flatten the tensors
        logits_flat = self.logits.view(-1, self.num_classes)
        labels_flat = self.labels.view(-1)

        # Filter out ignore_index
        valid_indices = labels_flat != self.ignore_index
        logits_valid = logits_flat[valid_indices]
        labels_valid = labels_flat[valid_indices]

        loss_ce = F.cross_entropy(logits_valid, labels_valid)

        # Assert that both losses are almost equal
        self.assertAlmostEqual(loss_smoothed.item(), loss_ce.item(), places=6,
                                msg=f"Loss with sigma=0 ({loss_smoothed.item()}) does not match standard CE loss ({loss_ce.item()})")

    def test_sigma_non_zero_label_smoother_changes_loss(self):
        """
        Test that GaussianLabelSmoother with sigma!=0 modifies the loss compared to standard cross entropy loss.
        """
        sigma = 1.0
        smoother = GaussianLabelSmoother(sigma=sigma, ignore_index=self.ignore_index, selector=None)

        # Compute loss with GaussianLabelSmoother
        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # Compute standard cross entropy loss ignoring the ignore_index
        # Flatten the tensors
        logits_flat = self.logits.view(-1, self.num_classes)
        labels_flat = self.labels.view(-1)

        # Filter out ignore_index
        valid_indices = labels_flat != self.ignore_index
        logits_valid = logits_flat[valid_indices]
        labels_valid = labels_flat[valid_indices]

        loss_ce = F.cross_entropy(logits_valid, labels_valid)

        # Assert that the losses are not almost equal
        # Allow some small difference due to numerical precision
        self.assertNotAlmostEqual(loss_smoothed.item(), loss_ce.item(), delta=1e-3,
                                msg=f"Loss with sigma={sigma} should differ from standard CE loss but they are similar ({loss_smoothed.item()} vs {loss_ce.item()})")

    def test_label_smoothing_with_ignore_index(self):
        """
        Test that GaussianLabelSmoother correctly ignores the ignore_index tokens.
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # Compute loss with GaussianLabelSmoother
        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # Manually compute the expected loss by ignoring the ignore_index
        logits_flat = self.logits.view(-1, self.num_classes)
        labels_flat = self.labels.view(-1)

        valid_indices = labels_flat != self.ignore_index
        logits_valid = logits_flat[valid_indices]
        labels_valid = labels_flat[valid_indices]

        # Manually create one-hot labels for sigma=1.0
        one_hot = F.one_hot(labels_valid, num_classes=self.num_classes).float()
        # Gaussian smoothing with sigma=1.0
        classes_arange = torch.arange(self.num_classes).unsqueeze(0)  # shape [1, num_classes]
        labels_flat_expanded = labels_valid.unsqueeze(1)  # shape [V, 1]
        dist_sq = (classes_arange - labels_flat_expanded) ** 2
        gauss = torch.exp(-dist_sq / (2 * (smoother.sigma ** 2)))
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)  # Normalize

        # Compute cross entropy with smoothed labels
        log_probs = F.log_softmax(logits_valid, dim=-1)
        loss_manual = -(gauss * log_probs).sum(dim=-1).mean()

        # Compare with loss_smoothed
        self.assertAlmostEqual(loss_smoothed.item(), loss_manual.item(), places=6,
                                msg=f"Smoothed loss ({loss_smoothed.item()}) does not match manually computed loss ({loss_manual.item()})")

    def test_no_valid_tokens_returns_zero_loss(self):
        """
        Test that if all labels are ignore_index, the loss returned is zero.
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # All labels are ignore_index
        labels_all_ignore = torch.full((self.batch_size, self.seq_length), self.ignore_index, dtype=torch.long)

        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, labels_all_ignore, shift_labels=False)

        # Expect the loss to be zero
        self.assertEqual(loss_smoothed.item(), 0.0, msg=f"Loss should be zero when all labels are ignore_index, but got {loss_smoothed.item()}")

    def test_selector_not_none(self):
        """
        Test that when a selector is provided, only selected tokens are smoothed.
        """
        # Define a mock selector that selects only the first token in each sequence
        class MockSelector:
            def __init__(self, num_classes, device):
                # Initialize nvocab with NaN for all classes
                self.nvocab = torch.full((num_classes,), float('nan'), device=device)
                # Set only the first token as a number token (f.e. here the decoded number is 0.0) 
                self.nvocab[0] = 0.0 # TODO: SET TO OTHER VALUE AND CHECK IF it STiLL WORKS!
            
            def select_number_tokens(self, logits, labels):
                number_tokens = torch.zeros_like(labels, dtype=torch.bool)
                number_tokens[:, 0] = True  # Select only the first token in each sequence
                return logits, labels, number_tokens

        # Instantiate the MockSelector with the appropriate number of classes and device
        selector = MockSelector(num_classes=self.num_classes, device=self.device)
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=selector)

        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # Manually compute the expected loss by selecting only the first token
        logits_selected = self.logits[:, 0, :]  # First token in each sequence
        labels_selected = self.labels[:, 0]      # Corresponding labels

        # Compute Gaussian labels manually
        classes_arange = torch.arange(self.num_classes, device=self.device).unsqueeze(0)  # Shape: [1, C]
        labels_flat_expanded = labels_selected.unsqueeze(1).float()                      # Shape: [B, 1]
        dist_sq = (classes_arange - labels_flat_expanded) ** 2
        gauss = torch.exp(-dist_sq / (2 * (smoother.sigma ** 2)))                          # Shape: [B, C]
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)                                   # Normalize

        # Compute cross entropy with smoothed labels
        log_probs = F.log_softmax(logits_selected, dim=-1)                                 # Shape: [B, C]
        loss_manual = -(gauss * log_probs).sum(dim=-1).mean()                            # Scalar

        # Compare with loss_smoothed
        self.assertAlmostEqual(
            loss_smoothed.item(),
            loss_manual.item(),
            places=6,
            msg=f"Smoothed loss with selector ({loss_smoothed.item()}) does not match manually computed loss ({loss_manual.item()})"
        )


    def test_sigma_zero_no_nan(self):
        """
        Test that when sigma=0, the loss is correctly computed and does not result in NaN.
        """
        smoother = GaussianLabelSmoother(sigma=0.0, ignore_index=self.ignore_index, selector=None)

        # Compute loss with GaussianLabelSmoother
        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # Ensure the loss is not NaN
        self.assertFalse(torch.isnan(loss_smoothed).any(),
                        msg=f"Loss with sigma=0 should not be NaN, but got {loss_smoothed.item()}")

    def test_sigma_zero_gradients(self):
        """
        Test that gradients can be computed when sigma=0.
        """
        smoother = GaussianLabelSmoother(sigma=0.0, ignore_index=self.ignore_index, selector=None)

        # Compute loss with GaussianLabelSmoother
        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # Backward pass
        loss_smoothed.backward()

        # Check that gradients are not None and have been computed
        self.assertIsNotNone(self.logits.grad,
                            msg="Gradients should be computed when sigma=0")
        self.assertFalse(torch.isnan(self.logits.grad).any(),
                            msg="Gradients should not contain NaN when sigma=0")

    def test_sigma_small_non_zero(self):
        """
        Test that a very small sigma (close to 0 but not 0) behaves similarly to sigma=0 but does not exactly match.
        """
        sigma = 1e-5
        smoother = GaussianLabelSmoother(sigma=sigma, ignore_index=self.ignore_index, selector=None)

        # Compute loss with GaussianLabelSmoother
        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # Compute standard cross entropy loss ignoring the ignore_index
        logits_flat = self.logits.view(-1, self.num_classes)
        labels_flat = self.labels.view(-1)

        valid_indices = labels_flat != self.ignore_index
        logits_valid = logits_flat[valid_indices]
        labels_valid = labels_flat[valid_indices]

        loss_ce = F.cross_entropy(logits_valid, labels_valid)

        # Assert that the losses are almost equal, as sigma is very small
        self.assertAlmostEqual(loss_smoothed.item(), loss_ce.item(), places=4,
                            msg=f"Loss with small sigma={sigma} ({loss_smoothed.item()}) should be close to standard CE loss ({loss_ce.item()})")

    def test_sigma_large(self):
        """
        Test that a very large sigma results in uniform label smoothing.
        """
        sigma = 100.0
        smoother = GaussianLabelSmoother(sigma=sigma, ignore_index=self.ignore_index, selector=None)

        # Compute loss with GaussianLabelSmoother
        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # For large sigma, Gaussian distribution approaches uniform distribution
        # So, the labels_to_calculate_loss should be approximately uniform
        # Compute manual loss with uniform labels
        valid_mask = (self.labels != self.ignore_index)
        num_valid = valid_mask.sum().item()
        if num_valid > 0:
            uniform_labels = torch.ones((num_valid, self.num_classes), device=self.logits.device) / self.num_classes
            logits_valid = self.logits.view(-1, self.num_classes)[valid_mask.view(-1)]
            log_probs = F.log_softmax(logits_valid, dim=-1)
            loss_manual = -(uniform_labels * log_probs).sum(dim=-1).mean()

            self.assertAlmostEqual(loss_smoothed.item(), loss_manual.item(), places=3,
                                msg=f"Loss with large sigma={sigma} ({loss_smoothed.item()}) should approach uniform label smoothing loss ({loss_manual.item()})")
        else:
            self.assertEqual(loss_smoothed.item(), 0.0, msg="Loss should be zero when there are no valid tokens")

    def test_varying_num_classes(self):
        """
        Test that GaussianLabelSmoother works correctly with different numbers of classes.
        """
        num_classes = 10
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # Create logits with a different number of classes
        logits_new = torch.randn(self.batch_size, self.seq_length, num_classes, requires_grad=True)
        model_output = {"logits": logits_new}
        loss_smoothed = smoother(model_output, self.labels, shift_labels=False)

        # Manually compute the expected loss
        labels_flat = self.labels.view(-1)
        valid_indices = labels_flat != self.ignore_index
        logits_valid = logits_new.view(-1, num_classes)[valid_indices]
        labels_valid = labels_flat[valid_indices]

        one_hot = F.one_hot(labels_valid, num_classes=num_classes).float()
        classes_arange = torch.arange(num_classes).unsqueeze(0)
        labels_flat_expanded = labels_valid.unsqueeze(1)
        dist_sq = (classes_arange - labels_flat_expanded) ** 2
        gauss = torch.exp(-dist_sq / (2 * (smoother.sigma ** 2)))
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)

        log_probs = F.log_softmax(logits_valid, dim=-1)
        loss_manual = -(gauss * log_probs).sum(dim=-1).mean()

        self.assertAlmostEqual(loss_smoothed.item(), loss_manual.item(), places=6,
                            msg=f"Loss with num_classes={num_classes} ({loss_smoothed.item()}) does not match manually computed loss ({loss_manual.item()})")

    def test_empty_labels_tensor(self):
        """
        Test that the smoother can handle an empty labels tensor without crashing.
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # Create an empty labels tensor
        labels_empty = torch.empty((0, self.seq_length), dtype=torch.long)

        # Create corresponding logits
        logits_empty = torch.randn(0, self.seq_length, self.num_classes, requires_grad=True)

        model_output = {"logits": logits_empty}
        loss_smoothed = smoother(model_output, labels_empty, shift_labels=False)

        # Expect the loss to be zero
        self.assertEqual(loss_smoothed.item(), 0.0, msg="Loss should be zero when labels tensor is empty")

    def test_labels_out_of_range(self):
        """
        Test that the smoother raises an error when labels contain indices >= num_classes.
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # Create labels with an out-of-range index
        labels_out_of_range = torch.tensor([
            [1, 5, self.ignore_index],  # 5 >= num_classes=5
            [0, self.ignore_index, 4]
        ])

        model_output = {"logits": self.logits}

        with self.assertRaises(RuntimeError):
            # F.one_hot will raise an error for labels >= num_classes
            smoother(model_output, labels_out_of_range, shift_labels=False)

    def test_gradients_when_loss_zero(self):
        """
        Test that gradients are not computed or are zero when the loss is zero (all labels ignored).
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # All labels are ignore_index
        labels_all_ignore = torch.full((self.batch_size, self.seq_length), self.ignore_index, dtype=torch.long)

        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, labels_all_ignore, shift_labels=False)

        # Backward pass
        loss_smoothed.backward()

        # Check that gradients are zero
        self.assertTrue(torch.all(self.logits.grad == 0),
                        msg="Gradients should be zero when loss is zero")

    def test_minimal_input(self):
        """
        Test that the smoother can handle minimal inputs (Minimal shape => (B=1, S=1, C=2)). 
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=-100, selector=None)

        logits = torch.tensor([[[0.0, 1.0]]])  # shape [1,1,2]
        labels = torch.tensor([[1]])           # shape [1,1]

        # No shift_labels
        loss = smoother(model_output={"logits": logits}, labels=labels, shift_labels=False)
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_large_batch_size(self):
        """
        Test that the smoother scales correctly with larger batch sizes.
        """
        larger_batch_size = 16
        seq_length = 10
        num_classes = 8
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # Create larger logits and labels
        logits_large = torch.randn(larger_batch_size, seq_length, num_classes, requires_grad=True)
        labels_large = torch.randint(0, num_classes, (larger_batch_size, seq_length))
        # Introduce some ignore_index
        labels_large[labels_large % 5 == 0] = self.ignore_index

        model_output = {"logits": logits_large}
        loss_smoothed = smoother(model_output, labels_large, shift_labels=False)

        # Ensure loss is computed
        self.assertIsInstance(loss_smoothed.item(), float, msg="Loss should be a float value")

    def test_all_tokens_valid(self):
        """
        Test that when all tokens are valid (no ignore_index), the loss is computed correctly.
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # All labels are valid
        labels_all_valid = torch.tensor([
            [1, 2, 3],
            [0, 4, 2]
        ])

        model_output = {"logits": self.logits}
        loss_smoothed = smoother(model_output, labels_all_valid, shift_labels=False)

        # Manually compute the expected loss
        logits_flat = self.logits.view(-1, self.num_classes)
        labels_flat = labels_all_valid.view(-1)

        one_hot = F.one_hot(labels_flat, num_classes=self.num_classes).float()
        classes_arange = torch.arange(self.num_classes).unsqueeze(0)  # shape [1, num_classes]
        labels_flat_expanded = labels_flat.unsqueeze(1)  # shape [V, 1]
        dist_sq = (classes_arange - labels_flat_expanded) ** 2
        gauss = torch.exp(-dist_sq / (2 * (smoother.sigma ** 2)))
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)  # Normalize

        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss_manual = -(gauss * log_probs).sum(dim=-1).mean()

        # Compare with loss_smoothed
        self.assertAlmostEqual(loss_smoothed.item(), loss_manual.item(), places=6,
                            msg=f"Smoothed loss ({loss_smoothed.item()}) does not match manually computed loss ({loss_manual.item()})")

    def test_gradient_flow(self):
        """
        Test that gradients flow correctly through the loss computation.
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=self.ignore_index, selector=None)

        # Create a simple scenario
        logits = torch.randn(1, 1, self.num_classes, requires_grad=True)
        labels = torch.tensor([[2]])

        model_output = {"logits": logits}
        loss_smoothed = smoother(model_output, labels, shift_labels=False)

        # Perform backward pass
        loss_smoothed.backward()

        # Check that gradients are not None
        self.assertIsNotNone(logits.grad, msg="Gradients should be computed")

        # Check that gradients are non-zero
        self.assertTrue(torch.any(logits.grad != 0),
                        msg="Gradients should not be zero")

    def test_shift_labels(self):
        """
        Test that label shifting works correctly.
        We'll create a scenario with (B=1, S=4, C=3) and verify that
        the smoother correctly shifts labels and logits, and that the computed loss matches manual computation.
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=-100, selector=None)

        logits = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0],  # token_0
                    [0.1, 0.2, 0.3],  # token_1
                    [1.0, 1.0, 1.0],  # token_2
                    [2.0, 2.0, 2.0],  # token_3
                ]
            ],
            dtype=torch.float,
        )  # shape [1, 4, 3]
        labels = torch.tensor([[0, 1, 2, 2]])  # shape [1,4]

        # Compute loss with shift_labels=True
        model_output_shifted = {"logits": logits}
        loss_shifted = smoother(model_output_shifted, labels, shift_labels=True)

        # Manually shift labels and logits
        shifted_labels = labels[:, 1:].contiguous()    # Shape: [1,3]
        shifted_logits = logits[:, :-1, :].contiguous()  # Shape: [1,3,3]

        # Manually compute the expected loss
        # Flatten the tensors
        labels_flat = shifted_labels.view(-1)          # Shape: [3]
        logits_flat = shifted_logits.view(-1, 3)      # Shape: [3,3]

        # Create valid mask (all labels are valid in this test case)
        valid_mask = labels_flat != -100
        labels_valid = labels_flat[valid_mask]         # Shape: [3]
        logits_valid = logits_flat[valid_mask]         # Shape: [3,3]

        # One-hot encode the valid labels
        one_hot = F.one_hot(labels_valid, num_classes=3).float()  # Shape: [3,3]

        # Compute Gaussian smoothing
        classes_arange = torch.arange(3).unsqueeze(0)            # Shape: [1,3]
        labels_expanded = labels_valid.unsqueeze(1)             # Shape: [3,1]
        dist_sq = (classes_arange - labels_expanded) ** 2        # Shape: [3,3]
        gauss = torch.exp(-dist_sq / (2 * (smoother.sigma ** 2)))  # Shape: [3,3]
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)        # Normalize to shape: [3,3]

        # Compute log probabilities
        log_probs = F.log_softmax(logits_valid, dim=-1)          # Shape: [3,3]

        # Compute manual loss
        loss_manual = -(gauss * log_probs).sum(dim=-1).mean()

        # Compare the smoother's loss with the manual loss
        self.assertAlmostEqual(
            loss_shifted.item(),
            loss_manual.item(),
            places=6,
            msg=f"Smoothed loss ({loss_shifted.item()}) does not match manually computed loss ({loss_manual.item()})"
        )

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)