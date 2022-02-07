from wordle import Game, MaxInfoAgent as Agent

import time

with open("../words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("../words_guesses.txt", "r") as guesses_file:
    guesses = guesses_file.read().splitlines()

g = Game(word="admin", verbose=True)

start = time.time()
agent = Agent(answers, guesses, mode="standard", first_guess="reast")
(
    final_guess,
    n_guesses,
) = agent.play(g)
end = time.time()

print(
    f"Solution: '{final_guess}' found with {n_guesses} guesses in {end-start:.2f} seconds"
)
