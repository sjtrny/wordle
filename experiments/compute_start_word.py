from wordle import get_numeric_representations, get_bin_counts, entropy
import numpy as np
import time

with open("../words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("../words_guesses.txt", "r") as guesses_file:
    guesses = guesses_file.read().splitlines()

start_time = time.time()

guesses_numba, _ = get_numeric_representations(guesses)
answers_numba, answers_char_counts = get_numeric_representations(answers)

bin_counts = get_bin_counts(
    guesses_numba,
    answers_numba,
    answers_char_counts,
)

# Compute Entropy
guesses_entropy = entropy(bin_counts)

# Sort by entropy
sort_idx = np.argsort(guesses_entropy)

print(guesses_entropy[sort_idx])

guesses_sorted = np.array(guesses)[sort_idx]
best_start_word = guesses_sorted[-1]

stop_time = time.time()

print(
    f"Optimal start word: {best_start_word}, computed in {stop_time-start_time:.2f} seconds."
)
