import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple
import math

class NGramBase:
    def __init__(self):
        """
        Initialize placeholders and config dictionary.
        """
        self.current_config = {}
        self.n = 1  # default
        # dictionaries to store counts
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set()  # track all seen words

    def method_name(self) -> str:
        """
        Return the name of the method from config.
        """
        return f"Method Name: {self.current_config['method_name']}"

    def update_config(self, config) -> None:
        """
        Override the current configuration.
        """
        self.current_config = config
        if 'n' in config:
            self.n = config['n']

    def preprocess(self, text: str) -> str:
        """
        Example: Lowercase and remove non-alphanumeric except spaces.
        Customize as needed.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]+", "", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        A simple whitespace-based tokenizer.
        """
        return text.split()

    def fixed_preprocess(self, text: str) -> str:
        """
        Provided in template: remove punctuation & lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Provided in template: simple split by whitespace.
        """
        return text.split()

    def prepare_data_for_fitting(self, data: List[str], use_fixed = False) -> List[List[str]]:
        """
        Provided in template. Convert each line of text into token list.
        """
        processed = []
        if not use_fixed:
            for text in data:
                processed.append(self.tokenize(self.preprocess(text)))
        else:
            for text in data:
                processed.append(self.fixed_tokenize(self.fixed_preprocess(text)))
        return processed

    def fit(self, data: List[List[str]]) -> None:
        """
        Build n-gram and (n-1)-gram context counts from tokenized data.
        """
        # Clear old counts
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set()

        for sentence in data:
            # Update vocabulary
            for w in sentence:
                self.vocab.add(w)

            # For an n-gram model, we often prepend (n-1) start tokens <s>
            # or do something similar. This example keeps it simple.
            # If you want to handle boundaries explicitly, add code for <s>, <\s>.
            for i in range(len(sentence) - self.n + 1):
                ngram_tuple = tuple(sentence[i:i+self.n])
                # context is the first n-1 tokens in this n-gram
                context_tuple = ngram_tuple[:-1]  # all but last

                self.ngram_counts[ngram_tuple] = self.ngram_counts.get(ngram_tuple, 0) + 1

                # Count the context
                if self.n > 1:
                    self.context_counts[context_tuple] = self.context_counts.get(context_tuple, 0) + 1

        # If n=1 (unigram), context_counts is not used or is trivial
        if self.n == 1:
            # For unigrams, each token is an ngram; context is empty
            self.context_counts[()] = sum(self.ngram_counts.values())

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        By default, raise NotImplementedError. Each smoothing method
        should override this or define its own approach.
        """
        raise NotImplementedError("Override this in the smoothing subclass")

    def perplexity(self, text: str) -> float:
        """
        Compute perplexity of text under the model.
        """
        tokens = self.tokenize(self.preprocess(text))
        N = len(tokens)
        if N == 0:
            return float('inf')  # or handle empty text differently

        log_prob_sum = 0.0

        for i in range(N - self.n + 1):
            ngram_tuple = tuple(tokens[i:i+self.n])
            context_tuple = ngram_tuple[:-1] if self.n > 1 else ()
            word = ngram_tuple[-1]

            p = self.ngram_probability(context_tuple, word)
            if p <= 0:
                # If probability is zero, perplexity is infinite
                return float('inf')
            log_prob_sum += math.log(p)

        # Perplexity = exp(-1/N * log_prob_sum)
        # Some define perplexity over entire n-grams, others over tokens
        # If you prefer dividing by (N - n + 1), that’s also used. Adjust as needed.
        return math.exp(-log_prob_sum / N)

if __name__ == "__main__":
    tester_ngram = NGramBase()
    test_sentence = "This, is a ;test sentence."
    processed = tester_ngram.preprocess(test_sentence)
    tokens = tester_ngram.tokenize(processed)
    print(tokens)
