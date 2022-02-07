from wordle import get_numeric_representations, get_bin_table, bin_table_to_counts
import numpy as np
from scipy.stats import entropy
from wordle import information_gain

with open("../words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("../words_guesses.txt", "r") as guesses_file:
    guesses = guesses_file.read().splitlines()


guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
answers_numba, answers_char_counts = get_numeric_representations(answers)

bin_table = get_bin_table(guesses_numba, answers_numba, answers_char_counts)

bin_counts = bin_table_to_counts(
    bin_table, np.full(len(guesses), True), np.full(len(answers), True)
)

e = entropy(bin_counts)
entropy_sorting = np.argsort(e)

ig = information_gain(bin_counts)
ig_sorting = np.argsort(ig)

print(np.array_equal(entropy_sorting, ig_sorting))
