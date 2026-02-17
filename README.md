# Context-Aware Spelling Correction

Noisy-channel spelling correction system using smoothed N-gram language models. Built for COL884 (Natural Language Processing) at IIT Delhi.

## Overview

This project implements a context-aware spelling corrector that combines a language model with an error model to fix context-dependent spelling mistakes (homophones, morphological variants, etc.). The system trains N-gram models (unigram through 5-gram) on a text corpus, generates correction candidates using edit distance, and ranks them using the noisy-channel formulation: `P(word|context) * P(error|true_word)`.

Achieves **88% accuracy** on a curated confusion set of context-sensitive spelling errors.

## Approach

**Language Model:** N-gram models with multiple smoothing techniques:
- Kneser-Ney smoothing
- Interpolated smoothing
- Add-k smoothing
- Good-Turing discounting
- Stupid Backoff

**Error Model:** Candidate generation via Damerau-Levenshtein edit distance (configurable max distance), scored with an edit distance penalty.

**Correction Pipeline:**
1. For each token, check if it exists in the vocabulary
2. If not, generate candidates within edit distance threshold
3. Score each candidate: `log P(candidate | context) + edit_distance_penalty`
4. Select the highest-scoring candidate

## Project Structure

```
├── config.py                # Hyperparameters for all smoothing methods
├── ngram.py                 # Base N-gram class (fitting, perplexity, tokenization)
├── smoothing_classes.py     # All smoothing implementations (NoSmooth, AddK, StupidBackoff, GoodTuring, Interpolation, KneserNey)
├── error_correction.py      # SpellingCorrector: candidate generation + ranking
├── scoring.py               # Evaluation metrics (WER, BLEU, F1, Levenshtein similarity)
├── grader.py                # End-to-end grading pipeline
└── predictions/             # Output predictions
```

## How to Run

Install dependencies:
```bash
pip install nltk numpy pandas editdistance
```

Place training data in `data/train1.txt` and `data/train2.txt`, and the test set in `data/misspelling_public.txt` (format: `correct_text&&corrupt_text` per line).

Run the grader:
```bash
python grader.py
```

Or run the corrector directly:
```bash
python error_correction.py
```

Smoothing method and N-gram order can be configured in `config.py`.

## Results

| Metric | Score |
|--------|-------|
| Accuracy (composite) | ~88% |

The composite score is a weighted combination of word error rate, F1, BLEU, Levenshtein similarity, and sequence similarity (see `scoring.py`).

## Tech Stack

- Python
- NLTK
- NumPy / Pandas
- editdistance
