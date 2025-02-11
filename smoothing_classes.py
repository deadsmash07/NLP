import numpy as np
import pandas as pd
from ngram import NGramBase
from config import no_smoothing, add_k, stupid_backoff, good_turing, interpolation, kneser_ney
import math
from typing import Tuple

class NoSmoothing(NGramBase):
    def __init__(self):
        super(NoSmoothing, self).__init__()
        self.update_config(no_smoothing)

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        MLE: count(context+word) / count(context).
        If context count = 0, return 1/|V| or 0, depending on your choice for OOV.
        """
        ngram_tuple = context + (word,)
        numerator = self.ngram_counts.get(ngram_tuple, 0)
        denominator = self.context_counts.get(context, 0)

        if denominator == 0:
            # If we never saw this context, you can choose to return 1/|V| or 0
            # to handle unseen context differently.
            return 0.0
        return numerator / denominator


class AddK(NGramBase):
    def __init__(self):
        super(AddK, self).__init__()
        self.update_config(add_k)
        self.k = None

    def fit(self, data):
        # Call parent fit to build counts
        super().fit(data)
        # Now we can store k from config
        self.k = self.current_config.get('k', 1.0)

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        Add-k smoothing:
        (count(context+word) + k) / (count(context) + k * |V|)
        """
        ngram_tuple = context + (word,)
        numerator = self.ngram_counts.get(ngram_tuple, 0) + self.k
        denominator = self.context_counts.get(context, 0) + self.k * len(self.vocab)
        return numerator / denominator


class StupidBackoff(NGramBase):
    def __init__(self):
        super(StupidBackoff, self).__init__()
        self.update_config(stupid_backoff)
        self.alpha = None

    def fit(self, data):
        super().fit(data)
        self.alpha = self.current_config.get('alpha', 0.4)

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        Stupid Backoff:
        If count(context+word) > 0, use that fraction.
        Otherwise, back off to n-1 gram * alpha.
        For n=3, if tri-gram is 0, use alpha * bigram probability; if that is 0, alpha^2 * unigram, etc.
        """
        return self._stupid_backoff_prob(context, word, self.n)

    def _stupid_backoff_prob(self, context: Tuple[str], word: str, level: int) -> float:
        """
        Recursively compute probabilities with backoff until we reach unigrams or get a non-zero count.
        """
        if level == 1:
            # unigram fallback
            # count of the single word / total tokens
            count_w = self.ngram_counts.get((word,), 0)
            total = self.context_counts.get((), 0)  # for unigrams
            if total == 0:
                return 0.0
            return count_w / total

        ngram_tuple = context + (word,)
        numerator = self.ngram_counts.get(ngram_tuple, 0)
        denominator = self.context_counts.get(context, 0)

        if denominator > 0 and numerator > 0:
            return numerator / denominator
        else:
            # back off (strip the first token from context)
            shorter_context = context[1:] if len(context) > 0 else ()
            return self.alpha * self._stupid_backoff_prob(shorter_context, word, level-1)


class GoodTuring(NGramBase):
    def __init__(self):
        super(GoodTuring, self).__init__()
        self.update_config(good_turing)
        # You will store additional structures to handle Good-Turing:
        # e.g. counts-of-counts

    def fit(self, data):
        super().fit(data)
        # build counts-of-counts, implement Good-Turing discounting

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        # Implement Good Turing logic here
        # Placeholder:
        return 0.000001


class Interpolation(NGramBase):
    def __init__(self):
        super(Interpolation, self).__init__()
        self.update_config(interpolation)
        self.lambdas = []

    def fit(self, data):
        super().fit(data)
        self.lambdas = self.current_config.get('lambdas', [0.5, 0.5])

        # If you want to build counts for lower-order n-grams as well, 
        # you might do that by building separate bigram, unigram counts, etc.

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        Example: linear interpolation of bigram/unigram if n=2
        P(word|context) = lambda_1 * P_MLE_bigram + lambda_2 * P_MLE_unigram
        (Generalize for higher n if needed.)
        """
        # For n=2, context is (w_{i-1}), so bigram = (context[0], word)
        bigram_tuple = context + (word,)
        bigram_count = self.ngram_counts.get(bigram_tuple, 0)
        bigram_context_count = self.context_counts.get(context, 0)

        if self.context_counts.get((), 0) == 0:
            return 0.0

        p_bigram = 0.0
        if bigram_context_count > 0:
            p_bigram = bigram_count / bigram_context_count

        # Unigram
        p_unigram = self.ngram_counts.get((word,), 0) / self.context_counts.get((), 1)

        # Weighted sum
        # e.g. lambdas = [lambda_unigram, lambda_bigram] or the reverse
        # check which order you prefer
        lambda_uni = self.lambdas[0]
        lambda_bi = self.lambdas[1]
        return lambda_uni * p_unigram + lambda_bi * p_bigram


class KneserNey(NGramBase):
    def __init__(self):
        super(KneserNey, self).__init__()
        self.update_config(kneser_ney)
        self.d = None  # discount parameter

    def fit(self, data):
        super().fit(data)
        self.d = self.current_config.get('discount', 0.75)
        # Additional counts required for Kneser-Ney

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        # Implement the Kneser–Ney logic here
        # Typically:
        # P_KN = max(count(c,w) - d, 0)/count(c) + d*(# of unique words that follow c)/count(c) * P_KN(lesser context, w)
        return 0.000001  # placeholder

if __name__=="__main__":
    ns = NoSmoothing()
    ns.method_name()