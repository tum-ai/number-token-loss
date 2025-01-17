import json


def create_clean_jsonl(input_file, output_file):
    """
    Reads the original MultiRC-style JSONL (with passage -> multiple questions -> answers)
    and writes a new JSONL file in the format:

    {
      "question": <string>,
      "answer": <string>
    }

    - "question": we combine the passage text and the question text into one.
    - "answer": we concatenate all correct answers (label=1) from that question.
    """

    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            # Parse the original record
            record = json.loads(line)
            passage_text = record["passage"]["text"]
            questions = record["passage"]["questions"]

            # For each question in this passage
            for q in questions:
                question_text = q["question"]
                answers = q["answers"]

                # Gather all the correct answers (label=1)
                correct_answers = [
                    ans["text"] for ans in answers
                    if ans.get("label", 0) == 1
                ]

                # If no correct answers, you could skip or store empty
                if not correct_answers:
                    final_answer = ""
                else:
                    # Join multiple correct answers with " | " or any delimiter
                    final_answer = " | ".join(correct_answers)

                # Build the "question" field by including passage + question
                combined_question = (
                    f"{passage_text.strip()}\n\n"
                    f"Question: {question_text.strip()}"
                )

                # The "answer" field (here, just the correct answers)
                out_record = {
                    "question": combined_question,
                    "answer": final_answer
                }

                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_path = "../../data/multirc/data/val.jsonl"
    output_path = "../../data/multirc/data/preprocessed/val_clean.jsonl"

    create_clean_jsonl(input_path, output_path)
    print(f"Finished writing to {output_path}")
