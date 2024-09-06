import json
import re
import os


def read_json(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object and append it to the list
            data.append(json.loads(line))

    for i in range(len(data)):
        yield data[i]


def preprocess_numbers(text: str):
    # replace whitespaces in number like 1 000 000 with 1000000
    number_whitespace_regex = r"(^|\s)((\d{1,3})( \d{3})+(\.\d+)?)"
    matches = re.findall(number_whitespace_regex, text)
    for match in matches:
        text = text.replace(match[1], match[1].replace(" ", ""))

    # replace commas in number like 1,000,000 with 1000000
    number_comma_regex = r"((\d{1,3})(,\d{3})+(\.\d+)?)"
    matches = re.findall(number_comma_regex, text)
    for match in matches:
        text = text.replace(match[0], match[0].replace(",", ""))

    # add zero before decimal point in number like .5 with 0.5
    number_decimal_regex = r"(\D)(\.\d+)"
    matches = re.findall(number_decimal_regex, text)
    for match in matches:
        text = text.replace(match[1], "0" + match[1])

    return text


def main():
    # print current dir
    print(os.getcwd())

    file_path = "data/"

    for file_name in ["train_t_clean.jsonl", "val_t_clean.jsonl", "test_clean.jsonl"]:
        for line in read_json(file_path + file_name):
            question = preprocess_numbers(line['question'])
            answer = preprocess_numbers(line['answer'])
            line['question'] = question
            line['answer'] = answer

            with open(file_path + "preprocessed/" + file_name, 'a') as file:
                file.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
