import json
import multiprocessing
import os
import random
from copy import deepcopy
from itertools import combinations

import numpy as np
import typer
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


lock = multiprocessing.Lock()


def paraphrase_chunk(data_chunk, model, output_file):
    augmented_questions_chunk = []

    # Open the output file inside the function in append mode
    with open(output_file, "a") as f:
        for sample in tqdm(data_chunk, total=len(data_chunk)):
            try:
                # Process the question using the model
                new_q = model(sample["question"])
                augmented_question = {"question": new_q, "answer": sample["answer"]}
                augmented_questions_chunk.append(augmented_question)

                # Write each result immediately to the file (safely with a lock)
                with lock:
                    f.write(json.dumps(augmented_question) + "\n")

            except Exception as e:
                print(f"Error processing sample: {e}")


def run_paraphrasing(
    data, n_proc, model, n_aug: int = 1, output_file: str = "train_aug.jsonl"
):
    for aug in tqdm(range(n_aug), desc=f"Looping all data {n_aug} times"):
        data_chunks = np.array_split(data, n_proc)

        with multiprocessing.Pool(processes=n_proc) as pool:
            # Pass the output file to each process along with the data chunk
            pool.starmap(
                paraphrase_chunk, [(chunk, model, output_file) for chunk in data_chunks]
            )

    print(f"Augmentation completed and saved to {output_file}")


def generate_chunk(data_chunk, model, output_file, num_samples):
    new_questions_chunk = []

    # Open the output file inside the function in append mode
    with open(output_file, "a") as f:
        # Randomly sample k combinations
        for sample1, sample2 in tqdm(
            random.sample(list(combinations(data_chunk, 2)), num_samples),
            total=num_samples,
        ):
            try:
                # Process the question using the model
                query = f"Sample 1: Question: {sample1['question']}, Answer: {sample1['answer']} - Sample 2: ..."
                output = model(query)
                question = output.split("Question:")[1].split("Answer:")[0].strip()
                answer = output.split("Answer:")[1]
                augmented_question = {"question": question, "answer": answer}
                new_questions_chunk.append(augmented_question)

                with lock:
                    f.write(json.dumps(augmented_question, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error processing sample: {e}")


def run_generating(
    data, n_proc, n_samples, model, output_file: str = "train_aug.jsonl"
):
    np.random.shuffle(data)
    data_chunks = np.array_split(data, n_proc)

    with multiprocessing.Pool(processes=n_proc) as pool:
        # Pass the output file to each process along with the data chunk
        pool.starmap(
            generate_chunk,
            [(chunk, model, output_file, n_samples // n_proc) for chunk in data_chunks],
        )

    print(f"Generation completed and saved to {output_file}")


app = typer.Typer()


def run(
    mode: str = typer.Option(
        "paraphrase",
        help="Mode of operation, options are: `paraphrase` and `generate`.",
    ),
    n_aug: int = typer.Option(
        1, help="Number of augmentation rounds. Only used for paraphrase mode"
    ),
    n_samples: int = typer.Option(
        1, help="Number of samples to generate. Only used for generate mode"
    ),
    output_file: str = typer.Option(
        "train_aug.jsonl", help="Output file for augmented data"
    ),
):
    if mode == "paraphrase":
        model = LLM(
            token=os.getenv("TOGETHER_API_KEY"),
            task_prompt="Rewrite this math task. Make sure to only paraphrase/reformulate trivial words that have nothing to do with task definition. Do NOT alter ANY of the numbers in the task.",
            temperature=0.7,
        )
    elif mode == "generate":
        model = LLM(
            token=os.getenv("TOGETHER_API_KEY"),
            task_prompt="You will see 2 examples from a math dataset for a NLP model. Each example has a question and an answer. Your task is to generate one similar example. Start the question with 'Question:' and the correct answer with 'Answer:'. Do not include newlines, unless after the text reply and before the #### followed only by the number response. Your example will be used as ground truth to train a NLP model so make sure to value correctness of the example higher than creativity. Limit response to 300 words MAX.",
            temperature=0.7,
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        )
    else:
        raise ValueError(f"Unknown mode {mode}")

    data_root = "data/grade-school-math/grade_school_math/data"
    with open(f"{data_root}/train_clean.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    n_proc = multiprocessing.cpu_count()

    if mode == "paraphrase":
        run_paraphrasing(
            data, n_proc, model, n_aug=n_aug, output_file=f"{data_root}/{output_file}"
        )
    else:
        run_generating(
            data, n_proc, n_samples, model, output_file=f"{data_root}/{output_file}"
        )


app.command()(run)
if __name__ == "__main__":
    app()
