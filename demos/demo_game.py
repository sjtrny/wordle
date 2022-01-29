from wordle import Game, MaxInfoStandardAgent

with open("../words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("../words_guesses.txt", "r") as guesses_file:
    guesses = guesses_file.read().splitlines()

g = Game(word="crimp", verbose=True)
agent = MaxInfoStandardAgent(answers, guesses, first_guess="arise")
(
    final_guess,
    _,
) = agent.play(g)
print(final_guess)
