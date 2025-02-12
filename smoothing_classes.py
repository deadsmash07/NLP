# smoothing_classes.py

from typing import Tuple
import math
from ngram import NGramBase
from config import (
    no_smoothing,
    add_k,
    stupid_backoff,
    good_turing,
    interpolation,
    kneser_ney
)

class NoSmoothing(NGramBase):
    def __init__(self):
        super().__init__()
        self.update_config(no_smoothing)

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        MLE = count(context, word) / count(context).
        """
        ngram_tuple = context + (word,)
        numerator = self.ngram_counts.get(ngram_tuple, 0)
        denominator = self.context_counts.get(context, 0)

        if denominator == 0:
            # unseen context => prob = 0 (or 1/|V| if you want)
            return 0.0
        return numerator / denominator


class AddK(NGramBase):
    def __init__(self):
        super().__init__()
        self.update_config(add_k)
        self.k = 1.0

    def fit(self, data):
        # first call parent to build counts
        super().fit(data)
        self.k = self.current_config.get("k", 1.0)

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        (count + k) / (context_count + k * vocab_size)
        """
        ngram_tuple = context + (word,)
        numerator = self.ngram_counts.get(ngram_tuple, 0) + self.k
        denominator = self.context_counts.get(context, 0) + self.k * len(self.vocab)
        return numerator / denominator


class StupidBackoff(NGramBase):
    def __init__(self):
        super().__init__()
        self.update_config(stupid_backoff)
        self.alpha = 0.4

    def fit(self, data):
        super().fit(data)
        self.alpha = self.current_config.get("alpha", 0.4)

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        If we have a non-zero count for (context, word), return MLE.
        Otherwise, back off to n-1 gram * alpha, recursively.
        """
        return self._backoff_prob(context, word, self.n)

    def _backoff_prob(self, context: Tuple[str], word: str, level: int) -> float:
        if level == 1:
            # Unigram fallback
            count_w = self.ngram_counts.get((word,), 0)
            total = self.context_counts.get((), 0)
            if total == 0:
                return 0.0
            return count_w / total

        ngram_tuple = context + (word,)
        numerator = self.ngram_counts.get(ngram_tuple, 0)
        denominator = self.context_counts.get(context, 0)

        if denominator > 0 and numerator > 0:
            return numerator / denominator
        else:
            # back off
            shorter_context = context[1:] if len(context) > 0 else ()
            return self.alpha * self._backoff_prob(shorter_context, word, level - 1)


class GoodTuring(NGramBase):
    """
    A simplified Good-Turing approach:
      - We'll compute counts of counts after we have ngram_counts.
      - p*(r) = (r+1) * n_(r+1) / (N * n_r)
      - For unseen (r=0), we use r=0 formula => p*(0) = n_1 / (N * n_0?) or a typical approximation
    """
    def __init__(self):
        super().__init__()
        self.update_config(good_turing)
        # For Good-Turing
        self.total_ngrams = 0
        self.count_of_counts = {}  # maps count -> how many n-grams have that count

        # Probability cache
        self.gt_prob_cache = {}

        # Probability mass for unseen
        self.p0 = 0.0

    def fit(self, data):
        super().fit(data)
        # Build counts-of-counts
        self._build_counts_of_counts()
        # Precompute probability for each possible count
        self._compute_good_turing_probs()

    def _build_counts_of_counts(self):
        self.count_of_counts = {}
        self.total_ngrams = 0

        for ngram_tuple, c in self.ngram_counts.items():
            self.count_of_counts[c] = self.count_of_counts.get(c, 0) + 1
            self.total_ngrams += c  # sum of all counts

    def _compute_good_turing_probs(self):
        """
        For each count r that we see, compute:
           p*(r) = (r+1)*n_{r+1} / (N * n_r)
        If n_{r+1} doesn't exist, we might approximate or skip.
        Also handle p*(0).
        """
        # maximum observed count
        max_count = max(self.count_of_counts.keys()) if self.count_of_counts else 0
        n1 = self.count_of_counts.get(1, 0)

        # For unseen n-grams (r=0), typical formula is:
        # p*(0) = n1 / (N * number_of_possible_ngrams)
        # We'll approximate number_of_possible_ngrams ~ len(self.vocab)^(n) if bigrams/trigrams, but that might be huge.
        # Alternatively, a simpler approach is p*(0) = n1 / (N_total_ngrams).
        # We'll store p0 just for convenience if we want to return a value for unseen.

        if self.total_ngrams > 0:
            self.p0 = n1 / (self.total_ngrams * 1.0)
        else:
            self.p0 = 0.0

        # Precompute p*(r) for each r up to max_count
        for r in range(1, max_count + 1):
            nr = self.count_of_counts.get(r, 0)
            nr_plus = self.count_of_counts.get(r + 1, 0)
            if nr == 0:
                continue
            # Good-Turing
            # p*(r) = (r+1)*n_{r+1}/(N * n_r)
            pr_star = (r + 1.0) * nr_plus / (self.total_ngrams * nr)
            self.gt_prob_cache[r] = pr_star

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        Compute the Good-Turing probability for the n-gram.
        1) Get the raw count.
        2) If it has count r>0, use p*(r).
        3) If it's unseen (r=0), return p0 or some fraction of p*(1).
        """
        ngram_tuple = context + (word,)
        r = self.ngram_counts.get(ngram_tuple, 0)
        if r == 0:
            # unseen
            return self.p0
        else:
            # if we have a cached GT probability for r, use it
            return self.gt_prob_cache.get(r, self.p0)


class Interpolation(NGramBase):
    """
    Example of linear interpolation for a bigram model:
      P_interpolated(w_i | w_{i-1})
      = lambda1 * P_unigram(w_i) + lambda2 * P_bigram(w_i | w_{i-1})
    """
    def __init__(self):
        super().__init__()
        self.update_config(interpolation)
        self.lambdas = [0.3, 0.7]  # default

    def fit(self, data):
        super().fit(data)
        self.lambdas = self.current_config.get("lambdas", [0.3, 0.7])

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        # assume n=2
        bigram_tuple = context + (word,)
        bigram_count = self.ngram_counts.get(bigram_tuple, 0)
        bigram_context_count = self.context_counts.get(context, 0)

        # MLE bigram
        if bigram_context_count > 0:
            p_bigram = bigram_count / bigram_context_count
        else:
            p_bigram = 0.0

        # Unigram
        unigram_count = self.ngram_counts.get((word,), 0)
        total_unigrams = self.context_counts.get((), 1)  # sum of all unigrams
        p_unigram = unigram_count / total_unigrams

        lambda1, lambda2 = self.lambdas
        return lambda1 * p_unigram + lambda2 * p_bigram


class KneserNey(NGramBase):
    """
    A simplified Kneser-Ney for n=2 or 3. We'll do the bigram or trigram version.
    The formula for bigram KN:
      P_KN(w_i|w_{i-1}) = max(count(w_{i-1}, w_i) - d, 0)/count(w_{i-1})
                          + d * (# of distinct w' s.t. (w_{i-1}, w') occurs) / count(w_{i-1})
                            * P_KN(w_i)
    Where P_KN(w_i) is the continuation probability:
      P_KN(w_i) = # of distinct w' s.t. (w', w_i) occurs / # of all bigram types
    For trigram, there's a recursive approach, etc.
    """
    def __init__(self):
        super().__init__()
        self.update_config(kneser_ney)
        self.d = 0.75
        # We'll store extra structures:
        self.unique_continuations = {}      # For the "continuation" counts
        self.lower_order_model = {}         # For P_{KN}(w) if n=2 or for bigram if n=3
        self.total_bigram_types = 0

    def fit(self, data):
        super().fit(data)
        self.d = self.current_config.get("discount", 0.75)
        # We need to build the unique continuation counts, etc.
        # If n=2 (bigram KN):
        if self.n == 2:
            self._build_bigram_kneser_ney()
        elif self.n == 3:
            self._build_trigram_kneser_ney()
        # etc.

    def _build_bigram_kneser_ney(self):
        """
        For bigram KN:
        - # distinct predecessors for w: Count how many distinct w_{i-1} exist for each w_i
        - # distinct successors for w_{i-1}: how many distinct w_i exist for each w_{i-1}.
        """
        # Count distinct w_{i-1} for each w_i
        distinct_predecessors = {}
        # Count distinct w_i for each w_{i-1}
        distinct_successors = {}

        for (w1, w2), c in self.ngram_counts.items():
            # update distinct successors
            distinct_successors[w1] = distinct_successors.get(w1, set())
            distinct_successors[w1].add(w2)

            # update distinct predecessors
            distinct_predecessors[w2] = distinct_predecessors.get(w2, set())
            distinct_predecessors[w2].add(w1)

        # Now build the continuation probability for each w:
        # P_continuation(w) = (# of distinct predecessors) / (total number of distinct bigrams)
        # The total distinct bigrams is just the size of self.ngram_counts.
        self.total_bigram_types = len(self.ngram_counts)
        for w in self.vocab:
            preds = distinct_predecessors.get(w, set())
            cont_prob = len(preds) / float(self.total_bigram_types) if self.total_bigram_types > 0 else 0.0
            self.lower_order_model[w] = cont_prob

        # We'll store distinct_successors to compute alpha factor
        self.unique_continuations = { w1: len(distinct_successors[w1]) for w1 in distinct_successors }

    def _build_trigram_kneser_ney(self):
        """
        For trigram we do a similar approach, but we rely on the bigram Kneser-Ney as the lower order model.
        ...
        For brevity, let's do a minimal approach.
        """
        # You can generalize from bigram approach, building "distinct successors" for each (w_{i-2}, w_{i-1}), etc.
        self._build_bigram_kneser_ney()  # so we can still use lower-order bigram continuation
        # Additional trigram logic could be placed here

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        if self.n == 2:
            return self._bigram_kn_prob(context, word)
        elif self.n == 3:
            return self._trigram_kn_prob(context, word)
        # fallback
        return 0.000001

    def _bigram_kn_prob(self, context: Tuple[str], word: str) -> float:
        # context is (w1,) for bigram
        if len(context) < 1:
            return 0.0
        w1 = context[0]
        bigram_count = self.ngram_counts.get((w1, word), 0)
        w1_count = self.context_counts.get((w1,), 0)

        # max(count - d, 0) / count(w1)
        left_part = max(bigram_count - self.d, 0) / w1_count if w1_count > 0 else 0.0

        # lambda = d * (# of distinct successors for w1) / count(w1)
        distinct_succ = self.unique_continuations.get(w1, 0)
        lambda_factor = (self.d * distinct_succ / w1_count) if w1_count > 0 else 0.0

        # continuation probability for 'word'
        cont_prob = self.lower_order_model.get(word, 0.0)

        return left_part + lambda_factor * cont_prob

    def _trigram_kn_prob(self, context: Tuple[str], word: str) -> float:
        """
        For trigram:
          P(w3|w1,w2) = max(count(w1,w2,w3) - d, 0)/count(w1,w2)
                        + d*(# of distinct w3 that appear with w1,w2)/count(w1,w2) * P_KN(w3|w2)
        We'll do a simplified version where P_KN(w3|w2) is computed by bigram Kneser–Ney.
        """
        if len(context) < 2:
            # fallback to bigram approach
            return self._bigram_kn_prob(context[-1:], word)

        w1, w2 = context[-2], context[-1]
        trigram_count = self.ngram_counts.get((w1, w2, word), 0)
        bigram_count = self.ngram_counts.get((w1, w2), 0)  # might or might not exist
        bigram_context_count = self.context_counts.get((w1, w2), 0)

        # main discount part
        left_part = 0.0
        if bigram_context_count > 0:
            left_part = max(trigram_count - self.d, 0) / bigram_context_count

        # # distinct w3 with (w1,w2)
        # We'll store how many distinct next-words appear with (w1,w2).
        # We can build a dictionary of sets in fit() if needed, but let's do a quick approach:
        # This is not the most efficient but illustrative:
        distinct_next = 0
        for (a, b, c), cnt in self.ngram_counts.items():
            if a == w1 and b == w2:
                distinct_next += 1
        lambda_factor = 0.0
        if bigram_context_count > 0:
            lambda_factor = (self.d * distinct_next) / bigram_context_count

        # lower order bigram probability
        # context for bigram is just (w2,)
        bigram_cont_prob = self._bigram_kn_prob((w2,), word)

        return left_part + lambda_factor * bigram_cont_prob


if __name__=="__main__":
    ns = NoSmoothing()
    ns.method_name()