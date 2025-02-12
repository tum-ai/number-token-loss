import json

from datasets import Dataset


# Define a function to read the text file and yield examples
def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        question = lines[i].strip()
        answer = lines[i + 1].strip()
        yield {'question': question, 'answer': answer}


def read_txt_cooking(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        example = json.loads(line.strip())
        if not isinstance(example, dict):
            raise ValueError("Each line must be a JSON object representing a dictionary")
        yield example


# Define the dataset loading function
def load_txt_dataset(file_path):
    return Dataset.from_generator(read_txt, gen_kwargs={'file_path': file_path})


def read_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object and append it to the list
            data.append(json.loads(line))

    for i in range(len(data)):
        yield data[i]


def load_json_dataset(file_path):
    return Dataset.from_generator(read_json, gen_kwargs={'file_path': file_path})
