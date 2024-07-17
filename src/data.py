from datasets import Dataset

# Define a function to read the text file and yield examples
def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        question = lines[i].strip()
        answer = lines[i+1].strip()
        yield {'question': question, 'answer': answer}

# Define the dataset loading function
def load_txt_dataset(file_path):
    return Dataset.from_generator(read_txt, gen_kwargs={'file_path': file_path})