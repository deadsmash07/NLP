import math
import re
import nltk
from nltk.tokenize import ToktokTokenizer
import string

# Download required NLTK data (if not already present)
nltk.download('punkt', quiet=True)

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between s1 and s2.
    """
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,        # deletion
                dp[i][j-1] + 1,        # insertion
                dp[i-1][j-1] + cost    # substitution
            )
    print(f"[Levenshtein] Distance between '{s1}' and '{s2}': {dp[m][n]}")
    return dp[m][n]

def get_kgrams(word: str, k: int = 2) -> set:
    """
    Returns the set of k-grams for the word.
    """
    kgrams = set()
    if len(word) < k:
        kgrams.add(word)
    else:
        for i in range(len(word) - k + 1):
            kgrams.add(word[i:i+k])
    return kgrams

def jaccard_coefficient(word1: str, word2: str, k: int = 2) -> float:
    """
    Computes the Jaccard coefficient between the k-gram sets of two words.
    """
    grams1 = get_kgrams(word1, k)
    grams2 = get_kgrams(word2, k)
    inter = len(grams1.intersection(grams2))
    union = len(grams1.union(grams2))
    return inter / union if union != 0 else 0.0

#############################################
# SimpleSpellingCorrector Class
#############################################

class SimpleSpellingCorrector:
    def __init__(self, vocab: set, k: int = 2, jaccard_threshold: float = 0.3):
        self.vocab = set(vocab)
        self.k = k
        self.jaccard_threshold = jaccard_threshold
        # Precompute hash indices for fast pattern lookup.
        self._build_wildcard_indices()

    def _build_wildcard_indices(self):
        """
        Precompute two indices:
          1. replacement_index: keys are patterns of the same length as the word,
             with one letter replaced by '$'.
          2. insertion_index: keys are patterns of length word+1,
             created by inserting '$' at every position.
        """
        self.replacement_index = {}
        self.insertion_index = {}
        for word in self.vocab:
            # Replacement patterns (same length as word)
            for i in range(len(word)):
                key = word[:i] + '$' + word[i+1:]
                self.replacement_index.setdefault(key, set()).add(word)
            # Insertion patterns (length is word length + 1)
            for i in range(len(word)+1):
                key = word[:i] + '$' + word[i:]
                self.insertion_index.setdefault(key, set()).add(word)
        # For debugging: Uncomment to see the indices.
        # print("Replacement Index:", self.replacement_index)
        # print("Insertion Index:", self.insertion_index)

    def _lookup_candidates(self, word: str) -> set:
        """
        Generate wildcard patterns for the input word and use the precomputed indices
        to quickly gather candidate words.
        """
        candidates = set()
        # Insertion patterns: patterns with length = len(word)+1.
        for i in range(len(word)+1):
            pattern = word[:i] + '$' + word[i:]
            if pattern in self.insertion_index:
                candidates.update(self.insertion_index[pattern])
        # Replacement patterns: patterns with length = len(word).
        for i in range(len(word)):
            pattern = word[:i] + '$' + word[i+1:]
            if pattern in self.replacement_index:
                candidates.update(self.replacement_index[pattern])
        return candidates

    def correct_word(self, word: str) -> str:
        """
        Correct a single word.
        - If the word is in the vocabulary, return it as is.
        - Otherwise, first attempt to retrieve candidates using wildcard (hash) lookup.
          If found, choose the candidate with the lowest Levenshtein distance.
        - If no candidates are found from the wildcard lookup, fall back to words
          with the same first letter and use k-gram (Jaccard) filtering, then pick the best.
        """
        # Return immediately if the word is correct.
        if word in self.vocab:
            print(f"'{word}' is in vocabulary; no correction needed.")
            return word

        # Step 1: Retrieve candidates quickly via hash indices.
        candidate_set2 = self._lookup_candidates(word)
        print(f"Candidates from hash-based wildcard lookup for '{word}': {candidate_set2}")

        if candidate_set2:
            best_candidate = None
            best_distance = None
            for cand in candidate_set2:
                dist = levenshtein_distance(word, cand)
                if best_candidate is None or dist < best_distance:
                    best_candidate = cand
                    best_distance = dist
            print(f"Selected candidate from wildcard lookup: {best_candidate} (distance: {best_distance})")
            return best_candidate
        else:
            # Fallback: candidate set from words with the same first letter.
            candidate_set1 = {w for w in self.vocab if w[0] == word[0]}
            filtered = []
            for cand in candidate_set1:
                jc = jaccard_coefficient(word, cand, self.k)
                if jc >= self.jaccard_threshold:
                    filtered.append((cand, jc))
            if filtered:
                filtered.sort(key=lambda x: (-x[1], levenshtein_distance(word, x[0])))
                best_candidate = filtered[0][0]
                print(f"Candidates from fallback (same first letter & k-gram): {filtered}")
                print(f"Selected candidate from fallback: {best_candidate}")
                return best_candidate
            else:
                print(f"No candidates found for '{word}' in fallback; returning original.")
                return word

    def correct_sentence(self, sentence: str) -> str:
        """
        Corrects each token in the sentence.
        Uses simple whitespace tokenization and preserves trailing punctuation.
        """
        tokens = sentence.split()
        corrected_tokens = []
        for token in tokens:
            # Separate trailing punctuation.
            punct = ""
            while token and not token[-1].isalnum():
                punct = token[-1] + punct
                token = token[:-1]
            corrected = self.correct_word(token.lower())
            # Preserve capitalization if needed.
            if token and token[0].isupper():
                corrected = corrected.capitalize()
            corrected_tokens.append(corrected + punct)
        corrected_sentence = " ".join(corrected_tokens)
        print(f"[Corrected Sentence]: {corrected_sentence}")
        return corrected_sentence

#############################################
# Demo Main Section
#############################################

if __name__ == "__main__":
    # ===========================
    # Step 1: Read and preprocess training corpus.
    # ===========================
    try:
        with open("./data/train1.txt", "r", encoding="utf-8") as f:
            training_text = f.read()
    except Exception as e:
        print("Error reading training file './data/train1.txt':", e)
        training_text = ""
    
    # Use NLTK's ToktokTokenizer to tokenize text with contraction preservation.
    tokenizer = ToktokTokenizer()
    def nltk_clean_tokenize(text):
        tokens = tokenizer.tokenize(text)
        # Remove tokens that do not contain any alphanumeric characters (e.g., punctuation-only tokens)
        filtered_tokens = [token for token in tokens if re.search(r'[A-Za-z0-9]', token)]
        return filtered_tokens

    training_tokens = nltk_clean_tokenize(training_text.lower())
    training_vocab = set(training_tokens)
    print(f"Training vocabulary size: {len(training_vocab)}")
    
    # If training vocabulary is empty, fallback to a default vocabulary.
    if not training_vocab:
        print("Training vocabulary is empty, using default vocabulary.")
        training_vocab = {
            "her", "here", "hair", "heart", "careful", "obstacles", "boardroom",
            "cat", "bat", "rat", "drat", "dart", "cart", "mare", "mane", "man",
            "maneuver", "semantic", "semaphore", "aboard", "border", "flew", "form",
            "from", "heathrow", "fled", "fore", "flea", "the", "rye", "catched",
            "static", "station", "statistics", "if", "you", "pick", "apples", "off",
            "of", "ground", "have", "to", "be", "wasps", "or", "else", "they", "will", "and", "comes", "up", "a", "big", "bump", "wherever", "it", "stungs"
        }
    
    # Instantiate the corrector with the training vocabulary.
    corrector = SimpleSpellingCorrector(training_vocab, k=2, jaccard_threshold=0.3)
    
    # ===========================
    # (Optional) Demo: Test a few words and sentences.
    # ===========================
    test_words = ["her", "hair", "carful", "obstcles"]
    for word in test_words:
        print("\n========================================")
        print(f"Original word: {word}")
        corrected = corrector.correct_word(word)
        print(f"Corrected word: {corrected}")

    test_sentences = [
        "If you pick apples off of the ground you have to be carful of the wass or else they will stung you.",
        "He ran around the obstcle quicly."
    ]
    for sentence in test_sentences:
        print("\n----------------------------------------")
        print("Original Sentence:", sentence)
        corrected_sentence = corrector.correct_sentence(sentence)
        print("Final Corrected Sentence:", corrected_sentence)
    
    # ===========================
    # Step 2: Read test file and process each line.
    # Test file format per line: <CORRECT TEXT> && <INCORRECT TEXT>
    # ===========================
    try:
        with open("./data/misspelling_public.txt", "r", encoding="utf-8") as f:
            test_lines = f.readlines()
    except Exception as e:
        print("Error reading test file './data/misspelling_public.txt':", e)
        test_lines = []

    for line in test_lines:
        if not line.strip():
            continue  # skip empty lines
        if "&&" not in line:
            print(f"Line in unexpected format: {line}")
            continue
        parts = line.split("&&")
        if len(parts) != 2:
            print(f"Line in unexpected format: {line}")
            continue
        correct_text, incorrect_text = parts
        # Correct the incorrect text using the correct_sentence method.
        corrected_sentence = corrector.correct_sentence(incorrect_text.strip())
        predicted_tokens = corrected_sentence.split()
        print("\n----------------------------------------")
        print("GT  :", correct_text.strip())
        print("IN  :", incorrect_text.strip())
        print("OUT :", " ".join(predicted_tokens))
