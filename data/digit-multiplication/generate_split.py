import random
from collections import deque
import os
import shutil

def get_last_n_lines(input_file_path, output_file_path, n):
    """
    Takes the last n lines from a JSONL file and writes them to another file.

    Parameters:
        input_file_path (str): Path to the input JSONL file.
        output_file_path (str): Path to the output file where the last n lines will be written.
        n (int): Number of lines to extract from the end of the file.
    """
    # Deque to store the last n lines
    last_n_lines = deque(maxlen=n)
    total_lines = 0

    # First pass: Read the input file and store the last n lines
    with open(input_file_path, 'r') as infile:
        for line in infile:
            last_n_lines.append(line)
            total_lines += 1

    # Calculate the number of lines to keep in the original file
    lines_to_keep = total_lines - n

    if lines_to_keep < 0:
        lines_to_keep = 0  # If n > total_lines

    # Second pass: Write the first lines_to_keep lines to a temp file
    temp_file_path = input_file_path + '.tmp'
    with open(input_file_path, 'r') as infile, open(temp_file_path, 'w') as temp_file:
        for _ in range(lines_to_keep):
            temp_file.write(next(infile))

    # Write the last n lines to the output file
    with open(output_file_path, 'a') as outfile:
        outfile.writelines(last_n_lines)

    # Replace the original file with the temp file
    shutil.move(temp_file_path, input_file_path)


def get_n_random_lines(input_file_path, output_file_path, n):
    """
    Takes n random lines from a JSONL file and writes them to another file.

    Parameters:
        input_file_path (str): Path to the input JSONL file.
        output_file_path (str): Path to the output file where the random lines will be written.
        n (int): Number of random lines to extract.
    """
    # First pass: Get total number of lines
    total_lines = 0
    with open(input_file_path, 'r') as infile:
        for _ in infile:
            total_lines += 1

    if n >= total_lines:
        n = total_lines  # If n >= total_lines, extract all lines

    # Randomly select n unique line numbers
    selected_line_numbers = set(random.sample(range(total_lines), n))

    # Second pass: Write selected and remaining lines to separate files
    temp_file_path = input_file_path + '.tmp'
    with open(input_file_path, 'r') as infile, \
            open(output_file_path, 'a') as outfile, \
            open(temp_file_path, 'w') as temp_file:
        for current_line_number, line in enumerate(infile):
            if current_line_number in selected_line_numbers:
                outfile.write(line)
            else:
                temp_file.write(line)

    # Replace the original file with the temp file
    shutil.move(temp_file_path, input_file_path)


if __name__ == "__main__":
    # print current path
    print(os.getcwd())

    input_file_path = "data/3_by_3_digit_fine_tune.jsonl"
    val_file_path = "data/val.jsonl"
    test_file_path = "data/test.jsonl"

    get_last_n_lines(input_file_path, test_file_path, 5000)
    get_last_n_lines(input_file_path, val_file_path, 3000)

    get_n_random_lines(input_file_path, test_file_path, 15000)
    get_n_random_lines(input_file_path, val_file_path, 7000)
