import json
from pathlib import Path

def tsv_to_jsonl(input_file, output_file):
    tsv_row_count = 0
    jsonl_row_count = 0
    with open(input_file, 'r', encoding='utf-8') as tsv_file:
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            for line in tsv_file:
                tsv_row_count += 1
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    number, text = parts
                    data = {"question": text, "answer": number}
                    jsonl_file.write(json.dumps(data) + '\n')
                    jsonl_row_count += 1
    if tsv_row_count == jsonl_row_count:
        print(f"Successfully processed {tsv_row_count} rows from {input_file} to {output_file}.")
    else:
        print(f"Warning: Mismatch in row counts! {tsv_row_count} rows in TSV, {jsonl_row_count} rows in JSONL.")


base_dir = Path(__file__).resolve().parent/'data'
file_names = ['train', 'dev', 'test']

tsv_files = [base_dir / f"{file_name}.tsv" for file_name in file_names]
jsonl_files = [base_dir / f"{file_name}.jsonl" for file_name in file_names]

# Convert each TSV file to JSONL
for tsv_file, jsonl_file in zip(tsv_files, jsonl_files):
    tsv_to_jsonl(tsv_file, jsonl_file)