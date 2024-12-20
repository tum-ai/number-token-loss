import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class ExpressionLoss:
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
        hashed_num_tokens = set(self.tokenizer.get_num_tokens())
        self.expression_tokens = self.tokenizer.expression_tokens

        self.start_token_id = self.tokenizer.convert_tokens_to_ids(
            self.expression_tokens[0]
        )
        self.end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.expression_tokens[1]
        )
        self.equal_token_id = self.tokenizer.convert_tokens_to_ids("=")
        self.operator_ids = self.tokenizer.convert_tokens_to_ids(["+", "-", "*"])
        self.underscore_id = self.tokenizer.convert_tokens_to_ids("_")

        # create a tensor of shape (vocab_size,) with the number tokens replaced by their corresponding number
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        for token, id in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                self.nvocab[id] = self.tokenizer.decode_number_token(
                    token, ignore_order=True
                )

        self.number_tokens = ~torch.isnan(self.nvocab)

    def convert_logit_seq_to_number(self, logits: Tensor):
        # extract weighted number according to probability
        softmaxed = F.softmax(logits, dim=-1)
        yhat_i = torch.sum(softmaxed * self.nvocab[self.number_tokens], dim=-1)

        # weight the predictions according to their decimal place
        reversed_y_hat_i = yhat_i.flip(0)
        powers_of_10 = torch.pow(10, torch.arange(reversed_y_hat_i.shape[-1]))
        y_hat = torch.sum(reversed_y_hat_i * powers_of_10)
        return y_hat

    def apply_operator(self, first_num, second_num, operator):
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

        # Create a mask for the expression
        start_locs = labels == self.start_token_id
        end_locs = labels == self.end_token_id
        equal_locs = labels == self.equal_token_id
        operator_locs = torch.isin(labels, torch.tensor(self.operator_ids))

        # do not compute a loss if no complete expression is found
        if (
            start_locs.sum() == 0
            or end_locs.sum() == 0
            or equal_locs.sum() == 0
            or operator_locs.sum() == 0
        ):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # extract expression locations
        batch_indices, start_indices = torch.where(start_locs)
        _, equal_indices = torch.where(equal_locs)
        _, end_indices = torch.where(end_locs)
        _, op_indices = torch.where(operator_locs)

        consistent_solution = []
        solution = []
        for batch, start, end, eq, op in zip(
            batch_indices, start_indices, end_indices, equal_indices, op_indices
        ):
            # extract operator
            operator_id = labels[batch, op]
            operator = self.tokenizer.convert_ids_to_tokens(operator_id.item())

            # extract label solutions
            solution_numbers = self.tokenizer.convert_ids_to_tokens(
                labels[batch, eq + 1 : end]
            )
            solution.append(float("".join(solution_numbers)))

            # extract predicted first number
            first_num = self.convert_logit_seq_to_number(
                logits[batch, start + 2 : op, :]
            )  # @TODO: Look at tokenization some reason the first token is always a '_' token.
            second_num = self.convert_logit_seq_to_number(logits[batch, op + 1 : eq, :])

            # extract predicted solution
            # pred_solution.append(self.convert_logit_seq_to_number(logits[batch, eq+1:end,:]))

            # compute result based on individual predicted numbers
            consistent_solution.append(
                self.apply_operator(first_num, second_num, operator)
            )

        loss = self.loss_function(
            torch.stack(consistent_solution), torch.tensor(solution)
        )

        return loss
