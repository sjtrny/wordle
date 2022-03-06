import pandas as pd
from wordle import MaxInfoSolver as Solver
import ast

with open("../words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("../words_guesses.txt", "r") as guesses_file:
    guesses = guesses_file.read().splitlines()


ngram_data = pd.read_csv("ngram_data.csv")

ngram_data['timeseries'] = ngram_data['timeseries'].apply(lambda x: ast.literal_eval(x)[-1])

# Check final play

codes = [
    "00100",
    "11000",
]

solution = "caulk"

solution_idx = answers.index(solution)

from wordle import mask_candidates, get_numeric_representations
import numpy as np

guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
answers_numba, answers_char_counts = get_numeric_representations(answers)



for code in codes:

    match_int = 0
    for idx, c in enumerate(code):
        match_int += int(c) << idx * 2

    # Eliminate answers that couldn't lead to the word

    total_mask = np.ones(len(answers)).astype(bool)

    total_mask = np.invert(mask_candidates(
        match_int,
        answers_numba[solution_idx, :],
        answers_numba,
        answers_char_counts,
        total_mask,
    ))

    print(np.count_nonzero(total_mask.astype(int)))

    remaining_guesses = np.array(answers)[total_mask]
    print(remaining_guesses)




#
# solver = Solver(answers, guesses, mode="standard")
#
# for play in plays[:-1]:
#
#     user_in_list = play.split(",")
#
#     guess = user_in_list[0]
#     code = user_in_list[1]
#
#     _, remaining_answers = solver.step(code, guess)
#
# # Rank words by n-gram
# ranked = ngram_data[ngram_data['ngram'].isin(remaining_answers)].sort_values(by='timeseries', ascending=False)
# print(ranked[['ngram', 'timeseries']])
#
# # print(remaining_answers)