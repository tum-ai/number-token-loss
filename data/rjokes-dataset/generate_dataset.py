import json
from pathlib import Path
from collections import Counter
from math import sqrt
import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error


def tsv_to_jsonl(input_file, output_file, distribution_output_file):
    tsv_row_count = 0
    jsonl_row_count = 0
    upvote_distribution = Counter() 

    with open(input_file, 'r', encoding='utf-8') as tsv_file:
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            for line in tsv_file:
                tsv_row_count += 1
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    number, text = parts
                    upvotes = int(number)  
                    upvote_distribution[upvotes] += 1  
                    data = {"question": text, "answer": upvotes}
                    jsonl_file.write(json.dumps(data) + '\n')
                    jsonl_row_count += 1

    if tsv_row_count == jsonl_row_count:
        print(f"Successfully processed {tsv_row_count} rows from {input_file} to {output_file}.")
    else:
        print(f"Warning: Mismatch in row counts! {tsv_row_count} rows in TSV, {jsonl_row_count} rows in JSONL.")
    with open(distribution_output_file, 'w', encoding='utf-8') as dist_file:
        json.dump(dict(upvote_distribution), dist_file, indent=4)
    print(f"Upvote distribution in {input_file}: {dict(upvote_distribution)}")

def calculate_errors(dev_file, train_mean):
    """
    Calculate RMSE, MSE, and MAE between actual upvotes and the train mean.
    """
    true_values = [] 
    predicted_values = [] 

    with open(dev_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            actual_upvotes = int(data["answer"]) 
            true_values.append(actual_upvotes)  # True upvote value
            predicted_values.append(train_mean) 

    y_true = np.array(true_values)
    y_pred = np.array(predicted_values)

    # Calculate errors using scikit-learn
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    return mse, rmse, mae


base_dir = Path(__file__).resolve().parent/'data'
file_names = ['train', 'dev', 'test']

tsv_files = [base_dir / f"{file_name}.tsv" for file_name in file_names]
jsonl_files = [base_dir / f"{file_name}.jsonl" for file_name in file_names]
distribution_files = [base_dir / f"{file_name}_distribution.json" for file_name in file_names]

# Convert each TSV file to JSONL
for tsv_file, jsonl_file, dist_file in zip(tsv_files, jsonl_files, distribution_files):
    tsv_to_jsonl(tsv_file, jsonl_file, dist_file)

with open(base_dir / "dev_distribution.json", 'r', encoding='utf-8') as dev_dist_file:
        train_distribution = json.load(dev_dist_file)
print(train_distribution)

train_total = sum(int(k) * v for k, v in train_distribution.items())  # Convert keys to integers
train_count = sum(train_distribution.values())  # Values are already integers
train_mean = train_total / train_count
print(f"Mean of train upvote distribution: {train_mean}")

# Calculate RMSE for the dev set using the train mean
dev_jsonl_file = base_dir / "dev.jsonl"
mse, rmse, mae = calculate_errors(dev_jsonl_file, train_mean)
# calculate_rmse(dev_jsonl_file, train_mean)

