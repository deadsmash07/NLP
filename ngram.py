# ngram.py
import re
import math
import nltk
import contractions
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple

# Download NLTK tokenizer model if needed
nltk.download('punkt', quiet=True)

class NGramBase:
    def __init__(self):
        """
        Base class for all n-gram language models.
        """
        self.current_config = {}
        self.n = 1  # default to unigram if not specified

        # N-gram count dictionaries
        self.ngram_counts = {}    # e.g., for bigram: dict((w1, w2) -> count)
        self.context_counts = {}  # e.g., for bigram: dict((w1,) -> count)
        self.vocab = set()        # unique vocabulary of words

    def update_config(self, config) -> None:
        """
        Override the current configuration.
        """
        self.current_config = config
        if "n" in config:
            self.n = config["n"]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize using NLTK's word_tokenize while matching the tokenization logic.
        """
        tokens = word_tokenize(text)

        # Reassemble common contractions
        contraction_suffixes = {"n't", "'s", "'m", "'re", "'ve", "'ll"}
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i+1] in contraction_suffixes:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        # Remove tokens that contain no alphanumeric characters
        filtered_tokens = [token for token in new_tokens if re.search(r'[A-Za-z0-9]', token)]
        
        return filtered_tokens

    def preprocess(self, text: str) -> str:
        """
        Enhanced preprocessing:
        - Expands contractions (e.g., "don't" → "do not")
        - Converts text to lowercase
        - Removes punctuation, except inside words (e.g., apostrophes and hyphens in words)
        """
        text = text.lower()
        
        # Expand contractions
        text = contractions.fix(text)

        # Remove symbols that are not part of words
        text = re.sub(r"(?<!\w)['\"“”‘’`´]|['\"“”‘’`´](?!\w)", "", text)  # Removes stray quotes
        text = re.sub(r"[^\w\s\-]", " ", text)  # Removes all symbols except hyphens in words
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        
        return text

    def prepare_data_for_fitting(self, data: List[str], use_fixed=False) -> List[List[str]]:
        """
        Converts each string in 'data' into a list of tokens.
        If use_fixed is True, do minimal tokenization, otherwise do full pipeline.
        """
        processed = []
        for text in data:
            if use_fixed:
                # Minimal: just split on whitespace
                tokens = text.lower().split()
            else:
                # Full pipeline
                cleaned = self.preprocess(text)
                tokens = self.tokenize(cleaned)
            processed.append(tokens)
        return processed

    def fit(self, data: List[List[str]]) -> None:
        """
        Build n-gram frequency counts from tokenized data.
        data is assumed to be a list of lists of tokens.
        """
        # Reset counts
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set()

        for sentence in data:
            # Update vocabulary
            self.vocab.update(sentence)

            # Count n-grams
            for i in range(len(sentence) - self.n + 1):
                ngram_tuple = tuple(sentence[i : i + self.n])
                context_tuple = ngram_tuple[:-1] if self.n > 1 else ()

                self.ngram_counts[ngram_tuple] = self.ngram_counts.get(ngram_tuple, 0) + 1
                if self.n > 1:
                    self.context_counts[context_tuple] = self.context_counts.get(context_tuple, 0) + 1

        # If n=1, store total count in context_counts[()]
        if self.n == 1:
            self.context_counts[()] = sum(self.ngram_counts.values())

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        Stub method. Each smoothing class will implement its version.
        """
        raise NotImplementedError("Override 'ngram_probability()' in a subclass.")

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

        # perplexity = exp(- (1/N) * sum of log p)
        return math.exp(-log_prob_sum / N)

    def method_name(self) -> str:
        """
        Return the name of the method from config.
        """
        return f"{self.current_config.get('method_name', 'UNKNOWN')}"

# Quick manual test
if __name__ == "__main__":
    tester = NGramBase()
    # Config example
    tester.update_config({"method_name": "BASELINE_TEST", "n": 2})

    data = [
        "The quick brown fox.",
        "The quick brown dog!",
        "The lazy dog sleeps..."
    ]
    prepared_data = tester.prepare_data_for_fitting(data)
    tester.fit(prepared_data)

    print("N-gram counts:", tester.ngram_counts)
    print("Context counts:", tester.context_counts)
    print("Vocabulary:", tester.vocab)
    print("Method Name:", tester.method_name())
