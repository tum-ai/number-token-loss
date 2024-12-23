import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class ExpressionLoss(object):
    def __init__(
        self,
        tokenizer: NumberEncodingTokenizer,
        vocab_size: int,
        device,
        loss_function=F.mse_loss,
        weight=0.5,
    ):
        self.tokenizer = tokenizer
        self.loss_function = loss_function
        self.weight = weight

        # Extract ids of all necessary tokens to generate expressions
        hashed_num_tokens = set(self.tokenizer.get_num_tokens())
        self.num_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.get_num_tokens()
        )
        self.expression_tokens = self.tokenizer.expression_tokens

        self.start_token_id = self.tokenizer.convert_tokens_to_ids(
            self.expression_tokens[0]
        )
        self.end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.expression_tokens[1]
        )
        self.equal_token_id = self.tokenizer.convert_tokens_to_ids("=")
        self.operator_ids = self.tokenizer.convert_tokens_to_ids(["+", "-", "*"])
        self.minus_id = self.tokenizer.convert_tokens_to_ids("-")
        self.neg_prec_ind = self.tokenizer.convert_tokens_to_ids(
            ["(", "▁(", "=", "▁", ">>"]
        )

        # Create a tensor of shape (vocab_size,) with the number tokens replaced by their corresponding number
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        for token, id in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                self.nvocab[id] = self.tokenizer.decode_number_token(
                    token, ignore_order=True
                )

        self.number_tokens = ~torch.isnan(self.nvocab)

    def extract_unary_negative_mask(
        self, negative_mask, preceding_bracket_locs, number_locs
    ):
        unary_negative_mask = (
            negative_mask
            & (
                torch.roll(
                    preceding_bracket_locs, shifts=1, dims=1
                )  # Preceded by a bracked index (dim=1: sequence)
            )
            & torch.roll(
                number_locs, shifts=-1, dims=1
            )  # Followed by a number (dim=1: sequence)
        )
        return unary_negative_mask

    def convert_logit_seq_to_number(self, logits: Tensor, labels: Tensor) -> Tensor:
        # Check if number is negative
        neg_factor = -1 if torch.isin(labels, self.minus_id).any() else 1

        # Reduce to number tokens only
        num_mask = torch.isin(labels, torch.tensor(self.num_token_ids))
        logits = logits[num_mask, :]

        # Extract weighted number according to probability
        softmaxed = F.softmax(logits, dim=-1)
        yhat_i = torch.sum(softmaxed * self.nvocab[self.number_tokens], dim=-1)

        # Weight the predictions according to their decimal place
        reversed_y_hat_i = yhat_i.flip(0)
        powers_of_10 = torch.pow(10, torch.arange(reversed_y_hat_i.shape[-1]))
        y_hat = neg_factor * torch.sum(reversed_y_hat_i * powers_of_10)
        return y_hat

    def apply_operator(
        self, first_num: Tensor, second_num: Tensor, operator: Tensor
    ) -> Tensor:
        if operator == "+":
            return first_num + second_num
        elif operator == "-":
            return first_num - second_num
        elif operator == "*":
            return first_num * second_num
        else:
            raise ValueError(f"Operator {operator} is not supported")

    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        labels = labels.masked_fill(labels == -100, 0)

        # Create a mask to filter out non-digit tokens
        logits = logits[:, :, self.number_tokens]

        # Create masks for all expression parts
        start_locs = labels == self.start_token_id
        end_locs = labels == self.end_token_id
        equal_locs = labels == self.equal_token_id
        operator_locs = torch.isin(labels, torch.tensor(self.operator_ids))
        number_locs = torch.isin(labels, torch.tensor(self.num_token_ids))
        negative_mask = labels == self.tokenizer.convert_tokens_to_ids("-")
        preceding_bracket_locs = torch.isin(labels, torch.tensor(self.neg_prec_ind))

        # Do not compute a loss if no complete expression is found
        if not (
            start_locs.any()
            or end_locs.any()
            or equal_locs.any()
            or operator_locs.any()
        ):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Refine operator_locs to exclude negative signs for numbers
        unary_negative_mask = self.extract_unary_negative_mask(
            negative_mask, preceding_bracket_locs, number_locs
        )
        operator_locs &= ~unary_negative_mask

        # Extract expression locations
        batch_indices, start_indices = torch.where(start_locs)
        _, equal_indices = torch.where(equal_locs)
        _, end_indices = torch.where(end_locs)
        _, op_indices = torch.where(operator_locs)

        # Check if the number of partial expressions found is consistent
        lengths = [
            len(start_indices),
            len(end_indices),
            len(equal_indices),
            len(op_indices),
        ]
        if set(lengths) != {lengths[0]}:
            raise ValueError("Mismatch in number of partial expressions found!")

        consistent_solution = []
        solution = []
        for batch, start, end, eq, op in zip(
            batch_indices, start_indices, end_indices, equal_indices, op_indices
        ):
            # Extract operator
            operator_id = labels[batch, op]
            operator = self.tokenizer.convert_ids_to_tokens(operator_id.item())

            # Extract label solutions
            solution_numbers = self.tokenizer.convert_ids_to_tokens(
                labels[batch, eq + 1 : end]
            )
            solution.append(float("".join(solution_numbers)))

            # Extract predicted first number
            first_num = self.convert_logit_seq_to_number(
                logits[batch, start + 2 : op, :], labels[batch, start + 2 : op]
            )  # @TODO: Look at tokenization some reason the first token is always a '_' token.
            second_num = self.convert_logit_seq_to_number(
                logits[batch, op + 1 : eq, :], labels[batch, op + 1 : eq]
            )

            # extract predicted solution
            # pred_solution.append(self.convert_logit_seq_to_number(logits[batch, eq+1:end,:]))

            # Compute result based on individual predicted numbers
            consistent_solution.append(
                self.apply_operator(first_num, second_num, operator)
            )

        loss = self.loss_function(
            torch.stack(consistent_solution), torch.tensor(solution)
        )

        return loss
