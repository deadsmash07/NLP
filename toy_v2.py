import re

def insertion_candidates_regex(input_word, vocab):
    # Build regex pattern parts for every possible insertion position.
    pattern_parts = []
    for i in range(len(input_word) + 1):
        # Escape parts of the word in case they contain regex special characters.
        part = re.escape(input_word[:i]) + "[A-Za-z]" + re.escape(input_word[i:])
        pattern_parts.append(part)
    # Combine with alternation.
    pattern = "^(?:" + "|".join(pattern_parts) + ")$"
    regex = re.compile(pattern)
    
    # Only consider words that are exactly one character longer.
    candidates = {word for word in vocab if len(word) == len(input_word) + 1 and regex.match(word)}
    return candidates

# Example usage:
vocab = {"grass", "griass", "gass", "class", "gassp", "classe"}
input_word = "gass"
print("Insertion candidates (via regex):", insertion_candidates_regex(input_word, vocab))
