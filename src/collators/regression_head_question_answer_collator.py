from typing import Dict, List, Union
import re
import torch
from transformers import DataCollatorForLanguageModeling


class RegressionHeadQuestionAnswerCLMCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        # Tokenize questions and answers separately
        questions = [example['question'] for example in examples]
        answers = [example['answer'] for example in examples]

        question_encodings = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")

        answer_numbers = []

        for answer in answers:
            answer_number = re.findall(r"\s*([+-]?\s*(\d+)(\.\d+)?)", answer)
            if not answer_number or len(answer_number) == 0:
                raise ValueError(f"Answer: {answer} does not contain any number")
            if len(answer_number) > 1:
                raise ValueError(f"Answer: {answer} contains more than one number")
            answer_numbers.append(float(answer_number[0][0]))

        answer_numbers = torch.tensor(answer_numbers, dtype=torch.float32).unsqueeze(1).to(question_encodings['input_ids'].device)

        input_ids = question_encodings['input_ids']
        attention_mask = question_encodings['attention_mask']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': answer_numbers
        }
