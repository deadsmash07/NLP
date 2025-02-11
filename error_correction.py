
import numpy as np
import pandas as pd

from typing import List
from smoothing_classes import (
    NoSmoothing, AddK, StupidBackoff, GoodTuring, Interpolation, KneserNey
)
from config import error_correction
import math

class SpellingCorrector:
    def __init__(self):
        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']

        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK()
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        # Update the chosen n-gram config
        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])

        # You can store additional error-model configs here too
        self.candidate_max_distance = self.correction_config.get("candidate_max_distance", 1)

    def fit(self, data: List[str]) -> None:
        """
        Fit the spelling corrector to the training data (for LM).
        """
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)

    def correct(self, text_tokens: List[str]) -> List[str]:
        """
        Correct each token in a list of tokens using a noisy-channel approach.
        For demonstration, we'll do a minimal approach:
          - If token is in vocab, keep it.
          - Otherwise, generate candidates, pick best by
            P_LM * error_model(prob).
        """
        corrected_tokens = []
        for idx, token in enumerate(text_tokens):
            if token in self.internal_ngram.vocab:
                # Keep it if recognized
                corrected_tokens.append(token)
            else:
                # Generate candidates
                candidates = self.generate_candidates(token)
                # Score each candidate
                best_cand = self.choose_best_candidate(text_tokens, idx, candidates)
                corrected_tokens.append(best_cand)
        return corrected_tokens

    def generate_candidates(self, token: str) -> List[str]:
        """
        Example: naive candidate generation by scanning vocab for small edit distance.
        This can be replaced with an advanced approach.
        """
        # For simplicity, return everything in the vocab (which is expensive if vocab is large!)
        # In practice, you'd filter by edit distance <= self.candidate_max_distance
        all_vocab = list(self.internal_ngram.vocab)
        return all_vocab

    def choose_best_candidate(self, text_tokens: List[str], idx: int, candidates: List[str]) -> str:
        """
        Score each candidate, picking the best.
        We approximate sentence probability by substituting the candidate for the token, 
        then measuring local n-gram probability, or the entire sentence probability. 
        If you have an explicit error model, incorporate that as well.
        """
        best_score = float('-inf')
        best_cand = text_tokens[idx]  # fallback

        for cand in candidates:
            # Build a local context for n-gram
            # For bigram, context = previous token
            # For trigram, context = previous 2 tokens, etc.
            # We can do a simpler approach: replace the token temporarily and measure perplexity
            original_token = text_tokens[idx]
            text_tokens[idx] = cand

            # Let's compute sum of log probabilities around that index only for speed
            score = self.local_ngram_score(text_tokens, idx)

            # If we had an error model, e.g., log(P(typo|cand)), multiply or add that:
            # score += math.log(self.error_model(token, cand))  # placeholder

            if score > best_score:
                best_score = score
                best_cand = cand

            # revert the token after scoring
            text_tokens[idx] = original_token

        return best_cand

    def local_ngram_score(self, tokens: List[str], idx: int) -> float:
        """
        Compute log probability for n-grams that include the token at index idx.
        This is more efficient than computing full sentence perplexity each time.
        For example, if n=2 (bigram), we only look at (token_{idx-1}, token_{idx}).
        If n=3, also look at (token_{idx-2}, token_{idx-1}, token_{idx}), etc.
        """
        log_prob = 0.0
        n = self.internal_ngram.n

        # For each relevant n-gram around idx:
        # For example, from (idx - n + 1) to idx, inclusive, if valid
        start = max(0, idx - n + 1)
        end = idx + 1  # Because range is exclusive at the upper bound
        for i in range(start, end):
            # build ngram from tokens[i : i + n] if in range
            if i + n <= len(tokens):
                ngram_tuple = tuple(tokens[i : i + n])
                context = ngram_tuple[:-1] if n > 1 else ()
                word = ngram_tuple[-1]
                p = self.internal_ngram.ngram_probability(context, word)
                if p > 0:
                    log_prob += math.log(p)
                else:
                    # If zero, it's often lethal for the candidate
                    return float('-inf')

        return log_prob
