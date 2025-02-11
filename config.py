# config.py

no_smoothing = {
    "method_name" : "NO_SMOOTH",
    "n": 2  # for example, bigram
}

add_k = {
    "method_name" : "ADD_K",
    "n": 2,
    "k": 1.0
}

stupid_backoff = {
    "method_name" : "STUPID_BACKOFF",
    "n": 3,
    "alpha": 0.4
}

good_turing = {
    "method_name" : "GOOD_TURING",
    "n": 2
}

interpolation = {
    "method_name" : "INTERPOLATION",
    "n": 2,
    "lambdas": [0.3, 0.7]  # for example
}

kneser_ney = {
    "method_name" : "KNESER_NEY",
    "n": 3,
    "discount": 0.75
}

error_correction = {
    "internal_ngram_best_config" : {
        "method_name" : "NO_SMOOTH",
        "n": 2,
    },

    # Additional hyperparams for your error model, candidate generation, etc.
    "candidate_max_distance": 1
}
