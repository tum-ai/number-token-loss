import re
import numpy as np

class XvalTokenizer:
    def __init__(self, pretrained_tokenizer=None, vocab_files=None, save_file=None, num_token="[NUM]"):
        if pretrained_tokenizer:
            self.tokenizer = pretrained_tokenizer
        else:
            assert vocab_files, 'Provide arg vocab_files (list of paths to training corpus files)'
            assert save_file, 'Provide arg save_file (path to save trained tokenizer)'
            self.tokenizer = get_tokenizer(vocab_files=vocab_files, save_file=save_file)
            
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [num_token],
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        })
        self.num_token = num_token
        self.num_token_id = self.tokenizer.convert_tokens_to_ids(num_token)
        print(f"Number token ID: {self.num_token_id}")

    def __call__(self, text, return_attention_mask=False, return_token_type_ids=True):
        if isinstance(text, dict):
            text = text["text"]

        nonum_text, numbers = extract(text, num_token=self.num_token)
        out = self.tokenizer(
            nonum_text, return_attention_mask=return_attention_mask, return_token_type_ids=return_token_type_ids
        )
        ids = np.array(out["input_ids"])
        locs = ids == self.num_token_id
        num_embed = np.ones(len(ids)).astype(np.float32)  # Use float32 instead of float16
        num_embed[locs] = numbers
        out["numbers"] = num_embed
        out["len"] = len(ids)
        return out


def extract(text, num_token="[NUM]"):
    import re

    # this regular expression is intended to match numerical values in various forms
    # like integers, floating-point numbers, or scientific notation, while avoiding
    # matching numbers that are part of strings.
    pattern = r"(?<!\')-?\d+(\.\d+)?([eE][-+]?\d+)?(?!\'|\d)"

    numbers = []

    def replace(match):
        numbers.append(match.group())
        return "¬"

    nonum_text = re.sub(pattern, replace, text)
    return compress_matrix(nonum_text).replace("¬", num_token), numbers


def compress_matrix(text):
    text = (
        text.replace("¬, ¬", "¬¬")
        .replace("¬, ¬", "¬¬")
        .replace("¬,¬", "¬¬")
        .replace("¬,¬", "¬¬")
    )
    return text

def get_tokenizer(vocab_files, save_file):
    """
    Train a Byte Pair Encoding (BPE) tokenizer and save it to a file.

    Args:
    vocab_files (list of str): List of paths to training corpus files.
    save_file (str): Path to save the trained tokenizer.

    Returns:
    Tokenizer: The trained tokenizer.
    """
    #special tokens (taken from xVal code)
    special_tokens = ["[END]", "[MASK]", "[PAD]", "[NUM]"]

    #train
    tokenizer = Tokenizer(models.BPE())
    tokenizer.add_special_tokens(special_tokens)
    
    full_vocab = []
    if vocab_files is not None:
        for file_path in vocab_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    full_vocab.extend(line.strip().split())

    trainer = trainers.BpeTrainer(vocab=full_vocab, special_tokens=special_tokens)
    tokenizer.train(vocab_files, trainer)

    #post-processing
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    # save 
    tokenizer.save(save_file)
    
    return tokenizer