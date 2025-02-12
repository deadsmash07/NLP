# config.py

# 1) NO SMOOTH
no_smoothing = {
    "method_name": "NO_SMOOTH",
    "n": 2
}

# 2) ADD-K
add_k = {
    "method_name": "ADD_K",
    "n": 2,
    "k": 1.0
}

# 3) STUPID BACKOFF
stupid_backoff = {
    "method_name": "STUPID_BACKOFF",
    "n": 3,
    "alpha": 0.4
}

# 4) GOOD TURING
good_turing = {
    "method_name": "GOOD_TURING",
    "n": 2
}

# 5) INTERPOLATION
interpolation = {
    "method_name": "INTERPOLATION",
    "n": 2,
    "lambdas": [0.3, 0.7]  # (unigram, bigram) for example
}

# 6) KNESER-NEY
kneser_ney = {
    "method_name": "KNESER_NEY",
    "n": 3,
    "discount": 0.75
}

# 7) ERROR CORRECTION
error_correction = {
    "internal_ngram_best_config": {
        "method_name": "NO_SMOOTH",
        "n": 2
    },
    # Additional parameters for the error model or candidate generation
    "candidate_max_distance": 3
}
