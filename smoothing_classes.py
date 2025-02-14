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
            # unseen context => prob = 0
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
      - For unseen (r=0), we use p*(0) = n_1 / (N_total_ngrams).
    """
    def __init__(self):
        super().__init__()
        self.update_config(good_turing)
        # For Good-Turing
        self.total_ngrams = 0
        self.count_of_counts = {}  # maps count -> how many n-grams have that count
        self.gt_prob_cache = {}
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
        max_count = max(self.count_of_counts.keys()) if self.count_of_counts else 0
        n1 = self.count_of_counts.get(1, 0)

        # p*(0) = n1 / (N_total_ngrams)
        if self.total_ngrams > 0:
            self.p0 = n1 / (self.total_ngrams * 1.0)
        else:
            self.p0 = 0.0

        # For each r in [1..max_count], p*(r) = (r+1)*n_{r+1}/(N * n_r)
        for r in range(1, max_count + 1):
            nr = self.count_of_counts.get(r, 0)
            nr_plus = self.count_of_counts.get(r + 1, 0)
            if nr == 0:
                continue
            pr_star = (r + 1.0) * nr_plus / (self.total_ngrams * nr)
            self.gt_prob_cache[r] = pr_star

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        ngram_tuple = context + (word,)
        r = self.ngram_counts.get(ngram_tuple, 0)
        if r == 0:
            # unseen
            return self.p0
        else:
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
    A simplified Kneser-Ney for n=2 or n=3.
    """
    def __init__(self):
        super().__init__()
        self.update_config(kneser_ney)
        self.d = 0.75
        # We'll store extra structures for Kneser-Ney:
        self.unique_continuations = {}
        self.lower_order_model = {}
        self.total_bigram_types = 0

    def fit(self, data):
        super().fit(data)
        self.d = self.current_config.get("discount", 0.75)
        if self.n == 2:
            self._build_bigram_kneser_ney()      # directly build from self.ngram_counts
        elif self.n == 3:
            self._build_trigram_kneser_ney()     # build separate bigram map from tri-grams

    def _build_bigram_kneser_ney(self):
        """
        Build Kneser-Ney structures for n=2 using self.ngram_counts which has pairs (w1, w2).
        """
        distinct_predecessors = {}
        distinct_successors = {}

        # self.ngram_counts is { (w1, w2): count, ... }
        for (w1, w2), c in self.ngram_counts.items():
            distinct_successors.setdefault(w1, set()).add(w2)
            distinct_predecessors.setdefault(w2, set()).add(w1)

        self.total_bigram_types = len(self.ngram_counts)

        # Continuation probability for each token
        for w in self.vocab:
            preds = distinct_predecessors.get(w, set())
            if self.total_bigram_types > 0:
                cont_prob = len(preds) / float(self.total_bigram_types)
            else:
                cont_prob = 0.0
            self.lower_order_model[w] = cont_prob

        # distinct successors for w1 (used for lambda factor)
        self.unique_continuations = {
            w1: len(distinct_successors[w1]) for w1 in distinct_successors
        }

    def _build_trigram_kneser_ney(self):
        """
        For n=3, we build a bigram-level model from the trigram counts.
        Then do the same distinct predecessor/successor logic.
        """
        # self.ngram_counts is { (w1, w2, w3): count, ... }

        # 1) Build a separate bigram map from these trigrams, so we can do bigram Kneser–Ney for the lower order.
        bigram_map = {}
        for (w1, w2, w3), count in self.ngram_counts.items():
            # The "lower order" bigram for Kneser-Ney is (w2, w3)
            bigram_map[(w2, w3)] = bigram_map.get((w2, w3), 0) + count

        # 2) Now do distinct successor/predecessor calculations on bigram_map
        distinct_predecessors = {}
        distinct_successors = {}

        for (b1, b2), c in bigram_map.items():
            distinct_successors.setdefault(b1, set()).add(b2)
            distinct_predecessors.setdefault(b2, set()).add(b1)

        self.total_bigram_types = len(bigram_map)

        for w in self.vocab:
            preds = distinct_predecessors.get(w, set())
            if self.total_bigram_types > 0:
                cont_prob = len(preds) / float(self.total_bigram_types)
            else:
                cont_prob = 0.0
            self.lower_order_model[w] = cont_prob

        self.unique_continuations = {
            b1: len(distinct_successors[b1]) for b1 in distinct_successors
        }

    def ngram_probability(self, context: Tuple[str], word: str) -> float:
        """
        The main entry point: Kneser-Ney probability for (context, word).
        """
        if self.n == 2:
            return self._bigram_kn_prob(context, word)
        elif self.n == 3:
            return self._trigram_kn_prob(context, word)
        # fallback for n>3 or n<1 (not implemented)
        return 1e-9

    def _bigram_kn_prob(self, context: Tuple[str], word: str) -> float:
        """
        P_KN(w | w1) for bigram. See standard Kneser-Ney formula:
          max(c(w1,w) - d, 0)/c(w1)  +  d*(# of distinct words after w1)/c(w1) * continuation_prob(w)
        """
        if len(context) < 1:
            return 0.0
        w1 = context[0]
        bigram_count = self.ngram_counts.get((w1, word), 0)
        w1_count = self.context_counts.get((w1,), 0)

        # left_part
        if w1_count > 0:
            left_part = max(bigram_count - self.d, 0) / w1_count
        else:
            left_part = 0.0

        # lambda factor
        distinct_succ = self.unique_continuations.get(w1, 0)
        if w1_count > 0:
            lambda_factor = (self.d * distinct_succ) / w1_count
        else:
            lambda_factor = 0.0

        # continuation probability from self.lower_order_model
        cont_prob = self.lower_order_model.get(word, 0.0)
        return left_part + lambda_factor * cont_prob

    def _trigram_kn_prob(self, context: Tuple[str], word: str) -> float:
        """
        P_KN(w | w1, w2) for trigram. 
        We discount the trigram counts. If none, back off to bigram Kneser-Ney.
        """
        if len(context) < 2:
            # fallback to bigram approach on last token of context
            return self._bigram_kn_prob(context[-1:], word)

        w1, w2 = context[-2], context[-1]
        trigram_count = self.ngram_counts.get((w1, w2, word), 0)
        bigram_context_count = self.context_counts.get((w1, w2), 0)

        # left part
        if bigram_context_count > 0:
            left_part = max(trigram_count - self.d, 0) / bigram_context_count
        else:
            left_part = 0.0

        # how many distinct next words follow (w1, w2)?
        distinct_next = 0
        # we can count how many unique 'word3' exist for that pair
        for (a, b, c) in self.ngram_counts:
            if a == w1 and b == w2:
                distinct_next += 1

        # lambda factor
        if bigram_context_count > 0:
            lambda_factor = (self.d * distinct_next) / bigram_context_count
        else:
            lambda_factor = 0.0

        # back off to bigram
        bigram_backoff = self._bigram_kn_prob((w2,), word)
        return left_part + lambda_factor * bigram_backoff



if __name__=="__main__":
    ns = NoSmoothing()
    ns.method_name()
