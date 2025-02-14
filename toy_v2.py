from spellchecker import SpellChecker

# Text containing the words
file="./data/train1.txt"
def read_file(file, mode='r'):
    with open(file, mode) as f:
        return f.read()

# Read file content
text = read_file(file)

# Extract words
words = set()
for line in text.split("\n"):
    words.update(line.replace(",", " ").replace("``", " ").split())

# Initialize spell checker
spell = SpellChecker()

# Identify misspelled words and get their correct suggestions
corrected_words = {word: spell.correction(word) for word in spell.unknown(words)}

# Save to file
output_file_path = "/corrected_words.txt"
with open(output_file_path, "w", encoding="utf-8") as f:
    for incorrect, correct in corrected_words.items():
        f.write(f"{incorrect} -> {correct}\n")

# Provide download link
print(f"Download corrected words file: [Download corrected words](sandbox:{output_file_path})")