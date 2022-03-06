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

plays = [
    "reast,00100",
    "align,11000",
    "caulk,22222"
]

solver = Solver(answers, guesses, mode="standard")

for play in plays[:-1]:

    user_in_list = play.split(",")

    guess = user_in_list[0]
    code = user_in_list[1]

    _, remaining_answers = solver.step(code, guess)

# Rank words by n-gram
ranked = ngram_data[ngram_data['ngram'].isin(remaining_answers)].sort_values(by='timeseries', ascending=False)
print(ranked[['ngram', 'timeseries']])

# print(remaining_answers)