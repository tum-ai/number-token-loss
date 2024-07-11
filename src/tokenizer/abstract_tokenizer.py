import re
import numpy as np
from abc import ABC, abstractmethod
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

class AbstractTokenizer(ABC):
    def __init__(self, special_tokens, num_tokens, embedding_dim, pretrained_tokenizer=None, vocab_files=None, save_file=None):
        if pretrained_tokenizer:
            self.tokenizer = pretrained_tokenizer
            self.tokenizer.add_special_tokens(special_tokens)
        else:
            if not vocab_files:
                raise ValueError('Provide arg vocab_files (list of paths to training corpus files)')
            if not save_file:
                raise ValueError('Provide arg save_file (path to save trained tokenizer)')
            self.tokenizer = self.get_tokenizer(vocab_files, save_file, special_tokens, num_tokens)
        self.num_tokens = num_tokens
        self.num_token_ids = [self.tokenizer.convert_tokens_to_ids(num_token) for num_token in num_tokens]
        self.embedding_dim = embedding_dim
        print(f"Number token IDs: {self.num_token_ids}")
        
    @abstractmethod
    def __call__(self, text, return_attention_mask=False, return_token_type_ids=True):
        pass

    @abstractmethod
    def extract(self, text):
        pass

    @staticmethod
    def get_tokenizer(vocab_files, save_file, special_tokens, num_tokens):

        tokenizer = Tokenizer(models.BPE())
        tokenizer.add_special_tokens(special_tokens)

        full_vocab = []
        for file_path in vocab_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    full_vocab.extend(line.strip().split())

        trainer = trainers.BpeTrainer(vocab=full_vocab, special_tokens=special_tokens)
        tokenizer.train(vocab_files, trainer)

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()

        tokenizer.save(save_file)
        
        return tokenizer
