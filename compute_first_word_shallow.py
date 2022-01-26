from wordle import get_numeric_representations, get_bin_counts, entropy
import numpy as np
import time

with open("words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("words_guesses.txt", "r") as guesses_file:
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
sort_idx = np.flip(np.argsort(guesses_entropy))

stop_time = time.time()

print(
    f"Optimal start word: {guesses[sort_idx[0]]}, computed in {stop_time-start_time:.2f} seconds."
)

f = open("first_guess_results_shallow.csv", "w")
f.write("guess,entropy\n")
for i in range(len(guesses)):
    f.write(f"{guesses[sort_idx[i]]},{guesses_entropy[sort_idx[i]]}\n")
f.close()