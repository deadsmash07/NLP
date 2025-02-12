# ngram.py

import re
import math
from typing import List, Dict, Tuple

class NGramBase:
    def __init__(self):
        """
        Base class for all n-gram language models.
        """
        self.current_config = {}
        self.n = 1  # by default
        # Store counts
        self.ngram_counts = {}
        self.context_counts = {}
        # Track vocabulary
        self.vocab = set()

    def update_config(self, config) -> None:
        """
        Override the current configuration.
        """
        self.current_config = config
        if "n" in config:
            self.n = config["n"]


    def preprocess(self, text: str) -> str:
        """
        Example: Lowercase and remove punctuation except spaces.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]+", "", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Splits text by whitespace.
        """
        return text.split()

    def fixed_preprocess(self, text: str) -> str:
        """
        Provided in the template: removes punctuation and converts to lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Provided in the template: simple split at spaces.
        """
        return text.split()
        self.vocab = set()  # track all seen words

    def method_name(self) -> str:
        """
        Return the name of the method from config.
        """
        return f"Method Name: {self.current_config['method_name']}"

    def prepare_data_for_fitting(self, data: List[str], use_fixed=False) -> List[List[str]]:
        """
        Converts each string in 'data' into a list of tokens.
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
        Build ngram_counts and context_counts from tokenized data.
        """
        # clear old counts
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set()

        for sentence in data:
            # update vocab
            for w in sentence:
                self.vocab.add(w)

            # Optionally, you can pad with (n-1) start tokens for boundary
            # For simplicity, we won't do that here.

            # count n-grams
            for i in range(len(sentence) - self.n + 1):
                ngram_tuple = tuple(sentence[i:i + self.n])
                context_tuple = ngram_tuple[:-1] if self.n > 1 else ()

                self.ngram_counts[ngram_tuple] = self.ngram_counts.get(ngram_tuple, 0) + 1
                if self.n > 1:
                    self.context_counts[context_tuple] = self.context_counts.get(context_tuple, 0) + 1

        # If n=1, the context is empty
        if self.n == 1:
            self.context_counts[()] = sum(self.ngram_counts.values())

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        Stub method. Each smoothing class will implement this.
        """
        raise NotImplementedError("Override 'ngram_probability()' in a smoothing subclass.")

    def perplexity(self, text: str) -> float:
        """
        Compute perplexity of the given text under the model.
        """
        tokens = self.tokenize(self.preprocess(text))
        N = len(tokens)
        if N == 0:
            return float("inf")

        log_prob_sum = 0.0
        for i in range(N - self.n + 1):
            ngram_tuple = tuple(tokens[i : i + self.n])
            context_tuple = ngram_tuple[:-1] if self.n > 1 else ()
            word = ngram_tuple[-1]
            p = self.ngram_probability(context_tuple, word)
            if p <= 0:
                return float("inf")
            log_prob_sum += math.log(p)

        # PP = exp(-1/N * (sum log p))
        return math.exp(-log_prob_sum / N)

if __name__ == "__main__":
    # Quick test
    tester = NGramBase()
    text = "Hello, world!"
    processed = tester.preprocess(text)
    tokens = tester.tokenize(processed)
    print(tokens)
