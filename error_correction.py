import math
from typing import List
from smoothing_classes import (
    NoSmoothing,
    AddK,
    StupidBackoff,
    GoodTuring,
    Interpolation,
    KneserNey
)
import editdistance
from config import error_correction

class SpellingCorrector:
    def __init__(self):
        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']

        smoothing_methods = {
            "NO_SMOOTH": NoSmoothing,
            "ADD_K": AddK,
            "STUPID_BACKOFF": StupidBackoff,
            "GOOD_TURING": GoodTuring,
            "INTERPOLATION": Interpolation,
            "KNESER_NEY": KneserNey
        }
        
        self.internal_ngram = smoothing_methods[self.internal_ngram_name]()
        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])

        # Candidate generation configuration
        self.candidate_max_distance = self.correction_config.get("candidate_max_distance", 2)

    def fit(self, data: List[str]) -> None:
        """
        Train the internal n-gram model on the provided text data.
        """
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)

    def correct(self, text_tokens: List[str]) -> List[str]:
        """
        Correct each token in a text by choosing the best alternative 
        using n-gram probability and an error model.
        """
        corrected_tokens = []
        for idx, token in enumerate(text_tokens):
            if token in self.internal_ngram.vocab:
                corrected_tokens.append(token)
            else:
                candidates = self.generate_candidates(token)
                best_candidate = self.choose_best_candidate(text_tokens, idx, candidates)
                corrected_tokens.append(best_candidate)
        return corrected_tokens

    def generate_candidates(self, token: str) -> List[str]:
        """
        Generate possible spelling corrections based on edit distance.
        """
        return [
            word for word in self.internal_ngram.vocab 
            if editdistance.eval(token, word) <= self.candidate_max_distance
        ]

    def choose_best_candidate(self, text_tokens: List[str], idx: int, candidates: List[str]) -> str:
        """
        Select the best candidate word based on a combination of:
        - **N-gram model probability**
        - **Edit distance penalty (error model)**
        """
        if not candidates:
            return text_tokens[idx]  # If no candidates, return the original token

        best_score = float("-inf")
        best_candidate = text_tokens[idx]

        original_token = text_tokens[idx]
        
        for candidate in candidates:
            text_tokens[idx] = candidate
            score = self.local_ngram_score(text_tokens, idx)
            
            # Error model: penalize by edit distance
            error_penalty = -editdistance.eval(original_token, candidate)

            total_score = score + error_penalty  # Combining LM probability and error penalty
            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate

        text_tokens[idx] = original_token  # Restore original token
        return best_candidate

    def local_ngram_score(self, tokens: List[str], idx: int) -> float:
        """
        Compute the local probability of a word in context using the n-gram model.
        """
        n = self.internal_ngram.n
        log_prob = 0.0

        start = max(0, idx - n + 1)
        end = min(idx + 1, len(tokens))

        for i in range(start, end):
            if i + n <= len(tokens):
                ngram_tuple = tuple(tokens[i : i + n])
                context_tuple = ngram_tuple[:-1] if n > 1 else ()
                word = ngram_tuple[-1]
                p = self.internal_ngram.ngram_probability(context_tuple, word)
                
                if p > 0:
                    log_prob += math.log(p)
                else:
                    return float("-inf")  # Prevents division errors

        return log_prob


if __name__ == "__main__":
    with open("data/train1.txt", "r") as f1, open("data/train2.txt", "r") as f2:
        train_data_1 = f1.read().splitlines()
        train_data_2 = f2.read().splitlines()
    
    train_data = train_data_1 + train_data_2

    corrector = SpellingCorrector()
    corrector.fit(train_data)

    # Uncomment to run evaluation with test data
    with open("data/misspelling_public.txt", "r") as f:
        for line in f:
            if "&&" in line:
                correct_text, incorrect_text = line.split("&&")
                incorrect_tokens = incorrect_text.strip().split()
                predicted_tokens = corrector.correct(incorrect_tokens)
                print("GT  :", correct_text.strip())
                print("IN  :", incorrect_text.strip())
                print("OUT :", " ".join(predicted_tokens))
                print()
