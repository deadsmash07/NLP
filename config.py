
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
    "alpha": 0.4 # 78.9 for alpha = 0.4 and n = 2 but 81.7 for n = 3
    # for n=4 and alpha = 0.4 gave 84.2
    # for n=6 and alpha = 0.4 gave 84.7
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
    "lambdas": [0.95, 0.05]  # (unigram, bigram) for example 0.793
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
        "method_name": "STUPID_BACKOFF",
        "n": 5,   
        "alpha": 0.4# 3 => trigram
        # "discount": 0.70  # Kneser-Ney discount
          # (unigram, bigram) for example

        
        
    },
    "candidate_max_distance": 3,  # max edit distance for isolated candidates
    "alpha": 5.0,                 # weight for normalized edit distance
    "beta": 1.0                   # weight for negative log n-gram probability
}

# error_correction = {
#     "internal_ngram_best_config": {
#         "method_name": "KNESER_NEY",
#         "n": 2,           # 3 => trigram--- gave 80% accuracy
#         "discount": 0.70  # Kneser-Ney discount
#     },
#     "candidate_max_distance": 3,  # max edit distance for isolated candidates
#     "alpha": 1.0,                 # weight for normalized edit distance
#     "beta": 0.1                   # weight for negative log n-gram probability
# }