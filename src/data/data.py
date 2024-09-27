from datasets import Dataset
import json
import os

from src.data.get_cooking_data import download_cooking

# Define a function to read the text file and yield examples
def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        question = lines[i].strip()
        answer = lines[i+1].strip()
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

def load_txt_cooking_dataset(file_path):
    return Dataset.from_generator(read_txt_cooking, gen_kwargs={'file_path': file_path})

def debug_cooking_dataset():
    if not os.path.exists("data/cooking_dataset_splits"):
        os.mkdir("data/cooking_dataset_splits")
        download_cooking()

    train_data_path = 'data/cooking_dataset_splits/train.txt'
    val_data_path = 'data/cooking_dataset_splits/val.txt'
    test_data_path = 'data/cooking_dataset_splits/test.txt'

    # Load the datasets
    train_dataset = load_txt_cooking_dataset(train_data_path)
    val_dataset = load_txt_cooking_dataset(val_data_path)
    test_dataset = load_txt_cooking_dataset(test_data_path)

    # Verify by printing the first example of each dataset
    print("Train Dataset:", train_dataset[0])  
    print("Validation Dataset:", val_dataset[0])  
    print("Test Dataset:", test_dataset[0])


def read_json(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object and append it to the list
            data.append(json.loads(line))

    for i in range(len(data)):
        yield data[i]

def load_json_dataset(file_path):
    return Dataset.from_generator(read_json, gen_kwargs={'file_path': file_path})

if __name__ == '__main__':
    debug_cooking_dataset()