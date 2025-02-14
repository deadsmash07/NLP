# error_correction.py
import math
import re
import nltk
import os
from typing import List, Set, Tuple

from nltk.tokenize import word_tokenize

# Import your smoothing classes and the config dictionary
from smoothing_classes import (
    NoSmoothing, AddK, StupidBackoff,
    GoodTuring, Interpolation, KneserNey
)
from config import error_correction

########################################################################
#                  1) HELPER FUNCTIONS
########################################################################

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between s1 and s2.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,     # deletion
                dp[i][j - 1] + 1,     # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def common_prefix_length(s1: str, s2: str) -> int:
    """
    Returns the number of characters that match from the start of s1 and s2.
    """
    l = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            l += 1
        else:
            break
    return l


def get_kgrams(word: str, k: int = 2) -> Set[str]:
    """
    Returns the set of k-grams for the word.
    """
    if len(word) < k:
        return {word}
    return {word[i : i + k] for i in range(len(word) - k + 1)}


def jaccard_coefficient(word1: str, word2: str, k: int = 2) -> float:
    """
    Computes the Jaccard coefficient between the k-gram sets of two words.
    """
    grams1 = get_kgrams(word1, k)
    grams2 = get_kgrams(word2, k)
    inter = len(grams1.intersection(grams2))
    union = len(grams1.union(grams2))
    return inter / union if union != 0 else 0.0


########################################################################
#             2) ISOLATED SPELLING CORRECTOR (with fallback)
########################################################################

class IsolatedSpellingCorrector:
    """
    A lightly adapted spelling corrector that:
      - uses wildcard-based candidate generation (insertion/replacement/deletion),
      - picks the best match by minimum Levenshtein distance (tie-break on first letter,
        then longest common prefix),
      - has a Jaccard-based fallback if the wildcard search finds no candidates.
    """
    def __init__(self, vocab: Set[str], k: int = 2, jaccard_threshold: float = 0.3, word_frequencies=None):
        self.vocab = set(vocab)
        self.k = k
        self.jaccard_threshold = jaccard_threshold
        self.word_frequencies = word_frequencies if word_frequencies else {} 
        self._build_wildcard_indices()
        
    def _build_wildcard_indices(self):
        """
        Precompute an index for replacement wildcards:
          Keys are patterns of the same length as the word with one letter replaced by '$'.
        """
        self.replacement_index = {}
        for word in self.vocab:
            for i in range(len(word)):
                key = word[:i] + '$' + word[i + 1:]
                self.replacement_index.setdefault(key, set()).add(word)

    def _insertion_candidates_regex(self, word: str) -> Set[str]:
        """
        Use a regex to find vocab words one letter longer that become 'word' when
        one letter is removed.
        """
        pattern_parts = []
        for i in range(len(word) + 1):
            # At position i, allow exactly one letter
            part = re.escape(word[:i]) + r"[A-Za-z]" + re.escape(word[i:])
            pattern_parts.append(part)
        pattern = "^(?:" + "|".join(pattern_parts) + ")$"
        regex = re.compile(pattern)

        # Only consider vocab words exactly one character longer
        return {
            w for w in self.vocab
            if len(w) == len(word) + 1 and regex.match(w)
        }

    def _lookup_candidates(self, word: str) -> Set[str]:
        """
        Generate candidate words that are within ~1 edit step:
          - insertion candidates (regex),
          - replacement patterns (wildcard),
          - deletion patterns (direct check in vocab).
        """
        candidates = set()

        # 1) Insertion
        insertion_candidates = self._insertion_candidates_regex(word)
        candidates.update(insertion_candidates)

        # 2) Replacement
        if len(word) > 1:  # at least length 2 so we can do a wildcard at position i
            for i in range(len(word)):
                key = word[:i] + '$' + word[i + 1:]
                if key in self.replacement_index:
                    candidates.update(self.replacement_index[key])

        # 3) Deletion (only if length > 2 or 3? can adapt your logic)
        if len(word) > 2:
            for i in range(len(word)):
                pattern = word[:i] + word[i + 1:]
                if pattern in self.vocab:
                    candidates.add(pattern)

        return candidates

    def correct_word_isolated(self, word: str) -> str:
        """
        Correct a single word, ignoring context. 
        - If word is in vocab, return it.
        - Else use wildcard-based lookup. If found, pick best by edit distance + tie-break.
        - Else fallback to Jaccard-based approach among words sharing the first letter.
        - If still none, return the original.
        """
        # Already correct
        if word in self.vocab:
            return word

        # 1) Wildcard-based candidate set
        candidate_set = self._lookup_candidates(word)
        if candidate_set:
            best_candidate = None
            best_distance = float('inf')
            for cand in candidate_set:
                dist = levenshtein_distance(word, cand)
                if dist < best_distance:
                    best_distance = dist
                    best_candidate = cand
                elif dist == best_distance:
                # Step 1: Prefer words that start with the same letter
                    if best_candidate and best_candidate[0] != word[0] and cand[0] == word[0]:
                        best_candidate = cand
                        continue  # Move to next candidate

                    # Step 2: Prefer words of the same length as the original word
                    if best_candidate and len(best_candidate) != len(word) and len(cand) == len(word):
                        best_candidate = cand
                        continue  # Move to next candidate

                    # Step 3: Use word frequency as the final tie-breaker
                    freq_cand = self.word_frequencies.get(cand, 0)
                    freq_best = self.word_frequencies.get(best_candidate, 0)

                    if freq_cand > freq_best:
                        best_candidate = cand

            return best_candidate

        # 2) Fallback: Jaccard with same first letter
        if not word:
            return word  # empty

        same_letter_candidates = {
            w for w in self.vocab
            if w and w[0] == word[0]
        }
        filtered = []
        for cand in same_letter_candidates:
            jc = jaccard_coefficient(word, cand, self.k)
            if jc >= self.jaccard_threshold:
                filtered.append((cand, jc))

        if filtered:
            # Sort by descending Jaccard, then ascending Levenshtein
            filtered.sort(key=lambda x: (-x[1], levenshtein_distance(word, x[0])))
            return filtered[0][0]

        # 3) If all fails, return original
        return word


########################################################################
#              3) CONTEXT-AWARE SPELLING CORRECTOR
########################################################################

class SpellingCorrector:
    """
    A combined approach that can do either:
      - purely isolated correction (if use_context=False),
      - or n-gram context-based re-ranking (if use_context=True).

    We also incorporate a list of frequent English words (from a txt file)
    into the isolated corrector's vocabulary, without including them in
    the n-gram model (they are not actually part of training sentences).
    """
    def __init__(self, use_context: bool = True, freq_words_path: str = "./data/frequent_words.txt"):
        self.use_context = use_context
        self.correction_config = error_correction  # from config.py
        self.internal_ngram_name = self.correction_config["internal_ngram_best_config"]["method_name"]

        # Instantiate the chosen n-gram model
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

        # Update the n-gram model with the chosen config
        self.internal_ngram.update_config(self.correction_config["internal_ngram_best_config"])

        # Will be assigned in fit()
        self.vocab = set()
        self.isolated_corrector = None

        # Weights (alpha=edit_distance, beta=ngram_probability)
        self.alpha = self.correction_config.get("alpha", 1.0)
        self.beta  = self.correction_config.get("beta", 1.0)

        # Path to frequent words file (one word per line)
        self.freq_words_path = freq_words_path

    def fit(self, data: List[str]) -> None:
        """
        Fit the n-gram model on 'data' (list of strings), then
        initialize the isolated corrector with the union of:
          - n-gram vocab
          - frequent English words loaded from a .txt file
        """
        # 1) Prepare training data and fit n-gram
        processed_data = self.internal_ngram.prepare_data_for_fitting(
            data,
            use_fixed=True
        )
        self.internal_ngram.fit(processed_data)

        # 2) Build the "true" vocabulary from the n-gram model
# 2) Build the "true" vocabulary from the n-gram model
        self.vocab = self.internal_ngram.vocab

        # 3) Compute word frequencies from the training corpus + extra corpus
        self.word_frequencies = {}
        for sentence in processed_data:
            for word in sentence:
                self.word_frequencies[word] = self.word_frequencies.get(word, 0) + 1

        # 4) Load frequent English words from external .txt file
        freq_words = set()
        if os.path.exists(self.freq_words_path):
            with open(self.freq_words_path, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip()
                    if w:
                        freq_words.add(w.lower())

        # 5) Combine vocab and frequent words
        combined_vocab = self.vocab.union(freq_words)

        # 6) Initialize the isolated corrector with word frequencies
        self.isolated_corrector = IsolatedSpellingCorrector(
            vocab=combined_vocab,
            k=2,
            jaccard_threshold=0.3,
            word_frequencies=self.word_frequencies  # Pass word frequency dictionary
        )


    def correct_tokens(self, tokens: List[str]) -> List[str]:
        """
        Correct a list of tokens by either:
          - purely isolated approach, or
          - context-based re-ranking (n-gram).
        """
        corrected_tokens = []
        n = self.internal_ngram.n  # e.g., 3 for trigram

        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # If no context usage, skip directly to the isolated corrector
            if not self.use_context:
                corrected = self.isolated_corrector.correct_word_isolated(token_lower)
                if token and token[0].isupper():
                    corrected = corrected.capitalize()
                corrected_tokens.append(corrected)
                continue

            # Else: context-based approach
            # Step A: gather candidate set from wildcard approach
            candidate_set = self.isolated_corrector._lookup_candidates(token_lower)

            # Also consider the possibility that the token itself is correct
            if token_lower in self.isolated_corrector.vocab:
                candidate_set.add(token_lower)

            # If still no candidates, fallback to isolated correct_word_isolated
            if not candidate_set:
                best_cand = self.isolated_corrector.correct_word_isolated(token_lower)
                if token and token[0].isupper():
                    best_cand = best_cand.capitalize()
                corrected_tokens.append(best_cand)
                continue

            # Step B: compute combined score (edit + ngram) for each candidate
            best_cand = None
            best_score = float("inf")

            # Build context window from already-corrected tokens
            if i >= (n - 1):
                context_window = corrected_tokens[i - (n - 1) : i]
            else:
                context_window = corrected_tokens[:i]

            # Lowercase for n-gram
            context_window = [c.lower() for c in context_window]
            context_tuple = tuple(context_window)

            for cand in candidate_set:
                # 1) Normalized edit distance
                edit_dist = levenshtein_distance(token_lower, cand)
                norm_edit = edit_dist / max(len(token_lower), len(cand))

                # 2) N-gram negative log probability
                p_ngram = self.internal_ngram.ngram_probability(context_tuple, cand)
                if p_ngram > 0:
                    ngram_log = -math.log(p_ngram)
                else:
                    ngram_log = 999.0  # big penalty for zero-prob

                # Weighted sum
                score = self.alpha * norm_edit + self.beta * ngram_log

                if score < best_score:
                    best_score = score
                    best_cand = cand

            # Preserve capitalization
            if token and token[0].isupper():
                best_cand = best_cand.capitalize()

            corrected_tokens.append(best_cand)

        return corrected_tokens

    def correct_text(self, text: str) -> str:
        """
        A convenience method to correct an entire sentence/string:
          1) Tokenize
          2) correct_tokens
          3) Reassemble
        """
        tokens = word_tokenize(text)
        corrected_tokens = self.correct_tokens(tokens)
        return " ".join(corrected_tokens)

    def correct(self, text: List[str]) -> List[str]:
        """
        The grader is calling: corrector.correct(self.corrupt)
        where 'self.corrupt' is a list of lines (strings).
        
        We must return a list of the same length, each being
        the corrected version of that line. This ensures the
        scoring logic in grader.py works as intended.
        """
        corrected_lines = []
        for line in text:
            corrected_line = self.correct_text(line)
            corrected_lines.append(corrected_line)
        
        # Now the length of corrected_lines == len(text).
        return corrected_lines


########################################################################
#              4) DEMO if run as __main__
########################################################################

if __name__ == "__main__":
    nltk.download('punkt', quiet=True)

    # Example training data
    sample_data = [
        "The quick brown fox jumps over the lazy dog",
        "The dog sleeps in the yard",
        "He wants to form a new plan from scratch"
    ]

    # 1) Create an instance (no context)
    corrector_isolated = SpellingCorrector(use_context=False)
    corrector_isolated.fit(sample_data)

    # 2) Test sentence with mistakes
    test_sentence = "Tha qyick broun fox jumpd ove the lazi dof. He wants to from a new plan."
    print("[Original]: ", test_sentence)
    corrected_isolated = corrector_isolated.correct_text(test_sentence)
    print("[Corrected] (isolated):", corrected_isolated)

    # 3) Now try context-based
    corrector_context = SpellingCorrector(use_context=True)
    corrector_context.fit(sample_data)
    corrected_context = corrector_context.correct_text(test_sentence)
    print("[Corrected] (context-based):", corrected_context)
