from wordle import StandardSolver

with open("../words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("../words_guesses.txt", "r") as guesses_file:
    guesses = guesses_file.read().splitlines()

solver = StandardSolver(answers, guesses)

print("-----------------")
print("WORDLE SOLVER")
print("-----------------")
print(
    "Code Format\n",
    "No Match (Gray): 0\n",
    "Partial Match (Yellow) 1\n",
    "Exact Match (Green): 2",
)
print("-----------------")

print(f"Optimal starting guess: reast")

while True:
    print("-----------------")
    user_in = input("Enter guess,code: ")
    if len(user_in) == 0:
        break

    user_in_list = user_in.split(",")

    try:
        guess = user_in_list[0]
        code = user_in_list[1]
        if not len(guess) == 5 and not len(code) == 5:
            raise Exception
    except:
        print("invalid input")
        continue

    suggested_guess, remaining_guesses = solver.step(code, guess)
    if (len(remaining_guesses)) > 0:
        print(
            f"Next guess: {suggested_guess}, {len(remaining_guesses)} valid guess words remaining",
        )
    else:
        print(f"Answer is {suggested_guess}")
        break
