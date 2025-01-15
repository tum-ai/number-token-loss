import random
from pathlib import Path

selected_categories = [
 
    "arithmetic__add_sub_multiple.txt",
   
]

train_folders = [
    "train-easy",
    "train-medium",
    "train-hard",
]

train_file_path = Path("arithmetic_train.txt")
val_file_path = Path("arithmetic_val.txt")

files = [Path("mathematics_dataset-v1.0", folder, category) for folder in train_folders for category in selected_categories]


length_data = 0
length_train = 0
length_val = 0

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()

        # Group every two lines (question, answer) together
        pairs = [(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]
        random.shuffle(pairs)  # Shuffle the pairs

        # Split the data
        train_end = 33333
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:(train_end+1000)]

        # Flatten the pairs back into a list of lines
        train_data = [line for pair in train_pairs for line in pair]
        val_data = [line for pair in val_pairs for line in pair]

        # Write to train and val files
        with open(train_file_path, 'a') as f:
            f.writelines(train_data)
        with open(val_file_path, 'a') as f:
            f.writelines(val_data)

        length_data += len(lines)
        length_train += len(train_data)
        length_val += len(val_data)

print(f"Data of size {length_data} has been split into train ({length_train} samples) and val ({length_val} samples).")


test_interpolate_folder = "interpolate"
test_interpolate_file_path = Path("arithmetic_test_interpolate.txt")

length_test_interpolate = 0

for category in selected_categories:
    file = Path("mathematics_dataset-v1.0", test_interpolate_folder, category)
    with open(file, 'r') as f:
        data = f.readlines()
        with open(test_interpolate_file_path, 'a') as f:
            f.writelines(data)

        length_test_interpolate += len(data)

print(f"Data of size {length_test_interpolate} has been written to arithmetic_test_interpolate.txt.")


test_extrapolate_folder = "extrapolate"
test_extrapolate_file_path = Path("arithmetic_test_extrapolate.txt")

length_test_extrapolate = 0
selected_categories = [
 
    "arithmetic__add_sub_multiple_longer.txt",
   
]

for category in selected_categories:
    file = Path("mathematics_dataset-v1.0", test_extrapolate_folder, category)
    with open(file, 'r') as f:
        data = f.readlines()
        with open(test_extrapolate_file_path, 'a') as f:
            f.writelines(data)

        length_test_extrapolate += len(data)

print(f"Data of size {length_test_extrapolate} has been written to arithmetic_test_extrapolate.txt.")
