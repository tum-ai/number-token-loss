from datasets import load_dataset, DatasetDict
import json

def download_cooking():
    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset('VincentLimbach/Cooking')

    # Function to split the dataset
    def split_dataset(dataset, train_size=0.7, val_size=0.15):
        # Shuffle the dataset
        dataset = dataset.shuffle(seed=42)
        
        # Calculate the split indices
        n = len(dataset)
        train_end = int(train_size * n)
        val_end = int((train_size + val_size) * n)
        
        # Split the dataset
        train_dataset = dataset.select(range(train_end))
        val_dataset = dataset.select(range(train_end, val_end))
        test_dataset = dataset.select(range(val_end, n))
        
        return train_dataset, val_dataset, test_dataset

    # Split each split of the dataset (assuming the dataset has a 'train' split)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset['train'])

    # Function to save datasets to text files
    def save_dataset_to_txt(dataset, file_path):
        with open(file_path, 'w') as f:
            for example in dataset:
                f.write(json.dumps(example) + '\n')

    # Save datasets to text files
    save_dataset_to_txt(train_dataset, 'data/cooking_dataset_splits/train.txt')
    save_dataset_to_txt(val_dataset, 'data/cooking_dataset_splits/val.txt')
    save_dataset_to_txt(test_dataset, 'data/cooking_dataset_splits/test.txt')

    print("Datasets have been saved to 'data/cooking_dataset_splits/'")


if __name__== '__main__':
    download_cooking()