from datasets import load_dataset
from pathlib import Path
import json

def clean_text(text):
    """Cleans text by removing newlines and ensuring proper Unicode decoding."""
    return text.replace("\n", " ")


def save_dataset(dataset_name="cnn_dailymail", dataset_version="3.0.0", data_dir="data"):
    """Loads and saves the CNN/DailyMail dataset as JSONL with cleaned text."""
    base_dir = Path(__file__).resolve().parent / data_dir
    base_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_version)

    # Define file paths
    file_names = ["train", "validation", "test"]
    jsonl_files = {split: base_dir / f"{split}.jsonl" for split in file_names}

    # Process and save each dataset split
    for split, jsonl_file in jsonl_files.items():
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for item in dataset[split]:
                data = {
                    "question": "summarize: " + clean_text(item["article"]),
                    "answer": clean_text(item["highlights"])
                }
                f.write(json.dumps(data) + "\n")

        print(f"Saved {split} dataset to {jsonl_file}")

def main():
    save_dataset()

if __name__ == "__main__":
    main()
