from datasets import Dataset
import json

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