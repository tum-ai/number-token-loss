from typing import Dict, List, Union
import torch
from transformers import DataCollatorForLanguageModeling
from src.utils.numerical_operations import signed_log


class RegressionHeadQuestionAnswerCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, log_scale: bool = False):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.log_scale = log_scale

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        # Tokenize questions and answers separately
        questions = [example['question'] for example in examples]
        answers = [example['answer'] for example in examples]

        question_encodings = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")

        answer_numbers = []

        for answer in answers:
            answer_numbers.append(float(answer))

        answer_numbers = torch.tensor(answer_numbers, dtype=torch.float32).unsqueeze(1).to(question_encodings['input_ids'].device)

        if self.log_scale:
            answer_numbers = signed_log(answer_numbers)

        input_ids = question_encodings['input_ids']
        attention_mask = question_encodings['attention_mask']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': answer_numbers
        }
