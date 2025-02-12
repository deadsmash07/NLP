import numpy as np
import editdistance
import nltk
from nltk.translate.bleu_score import sentence_bleu
from difflib import SequenceMatcher
from typing import List, Tuple

def word_error_rate(prediction_words: List[str], ground_truth_words: List[str]) -> float:
    edits = editdistance.eval(prediction_words, ground_truth_words)
    return edits / max(1, len(ground_truth_words))

def precision_recall_f1(prediction_words: List[str], ground_truth_words: List[str]) -> Tuple[float,float,float]:
    tp = sum(1 for w in prediction_words if w in ground_truth_words)
    fp = sum(1 for w in prediction_words if w not in ground_truth_words)
    fn = sum(1 for w in ground_truth_words if w not in prediction_words)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def levenshtein_similarity(prediction_string: str, ground_truth_string:str) -> float:
    distance = editdistance.eval(prediction_string, ground_truth_string)
    max_length = max(len(prediction_string), len(ground_truth_string))
    return 1 - (distance / max_length) if max_length > 0 else 1

def sentence_similarity(prediction_string: str, ground_truth_string: str) -> float:
    return SequenceMatcher(None, prediction_string, ground_truth_string).ratio()

def score(prediction_string: str, ground_truth_string: str) -> float:
    prediction_words = prediction_string.split()
    ground_truth_words = ground_truth_string.split()

    wer = word_error_rate(prediction_words, ground_truth_words)
    precision, recall, f1 = precision_recall_f1(prediction_words, ground_truth_words)
    bleu = sentence_bleu([ground_truth_words], prediction_words)
    levenshtein_sim = levenshtein_similarity(prediction_string, ground_truth_string)
    sequence_sim = sentence_similarity(prediction_string, ground_truth_string)

    score = (
        (1 - wer) * 0.3 + f1 * 0.2 + bleu * 0.2 +levenshtein_sim * 0.2 +sequence_sim * 0.1
    )

    return score

def score_batch(predictions: List[str], ground_truths: List[str]):
    return np.mean([score(p, g) for p, g in zip(predictions, ground_truths)])

if __name__ == "__main__":
    prediction = "tis is a shample text"
    ground_truth = "this is a sample text"
    print(score(prediction, ground_truth))