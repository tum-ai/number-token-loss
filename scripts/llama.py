import json
import multiprocessing
import os
from copy import deepcopy

import numpy as np
import requests
from together import Together
from tqdm import tqdm


class LLM:
    def __init__(
        self,
        token: str,
        task_prompt: str,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature: float = 0.7,
    ):
        self.token = token
        self.temperature = temperature
        self.task = task_prompt
        self.message_history = [{"role": "system", "content": task_prompt}]
        self.model = model
        self.client = Together(api_key=token)
        self.counter = 0

    def _add_to_message_history(self, role: str, content: str):
        self.message_history.append({"role": role, "content": content})

    def send_message(self, message: str, history: bool = True):
        # Add user's message to the conversation history.
        self._add_to_message_history("user", message)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.message_history,
            stream=True,
            temperature=self.temperature,
        )

        # Process and stream the response.
        response_content = ""
        self.counter += 1

        if not history:
            self.message_history = self.message_history[:-1]

        for token in response:
            delta = token.choices[0].delta.content
            # End token indicating the end of the response.
            if token.choices[0].finish_reason:
                if history:
                    self._add_to_message_history("assistant", response_content)
                break
            else:
                # Append content to message and stream it.
                response_content += delta
                yield delta

    def __call__(self, *args, **kwargs):
        response = self.send_message(*args, **kwargs)
        full_text = "".join([part for part in response])
        return full_text


def augment_chunk(data_chunk, model):
    augmented_questions_chunk = []
    for sample in tqdm(data_chunk, total=len(data_chunk)):
        new_q = model(sample["question"])
        augmented_questions_chunk.append(
            {"question": new_q, "answer": sample["answer"]}
        )
    return augmented_questions_chunk


def run_augmentation(data, n_proc, model, n_aug: int = 1):
    augmented_questions = []
    for aug in tqdm(range(n_aug), desc=f"Looping all data {n_aug} times"):
        data_chunks = np.array_split(data, n_proc)

        with multiprocessing.Pool(processes=n_proc) as pool:
            results = pool.starmap(
                augment_chunk, [(chunk, model) for chunk in data_chunks]
            )
        for result in results:
            augmented_questions.extend(result)

    return augmented_questions


if __name__ == "__main__":
    model = LLM(
        token=os.getenv("TOGETHER_API_KEY"),
        task_prompt="Rewrite this math task. Make sure to only paraphrase/reformulate trivial words that have nothing to do with task definition. Do NOT alter ANY of the numbers in the task.",
        temperature=0.7,
    )

    with open(
        "data/grade-school-math/grade_school_math/data/train_clean.jsonl", "r"
    ) as f:
        data = [json.loads(line) for line in f]

    n_proc = multiprocessing.cpu_count()

    augmented_questions = run_augmentation(data, n_proc, model)
    print(f"Generated {len(augmented_questions)} new questions")

    with open(
        "data/grade-school-math/grade_school_math/data/train_aug.jsonl", "a"
    ) as f:
        for q in augmented_questions:
            f.write(json.dumps(q) + "\n")
