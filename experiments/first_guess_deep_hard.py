from wordle import Game, HardAgent
from wordle import get_numeric_representations, get_bin_counts, entropy
import numpy as np
from multiprocessing import Pool, cpu_count
import time


def job(answer, first_guess, answers, guesses):
    g = Game(word=answer, verbose=False)
    agent = HardAgent(answers, guesses, first_guess=first_guess)
    (
        final_guess,
        n_guesses,
    ) = agent.play(g)
    if final_guess == answer:
        return answer, n_guesses
    else:
        return answer, np.nan


if __name__ == "__main__":
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
    sort_idx = np.argsort(guesses_entropy)
    guesses_sorted = np.flip(np.array(guesses)[sort_idx])

    pool = Pool(cpu_count())

    # How many guesses to check
    n_guesses = 100

    n_plays = np.zeros((len(answers), n_guesses))

    for i in range(n_guesses):
        print(f"{i+1}/{n_guesses}", guesses_sorted[i])
        results = pool.starmap(
            job, ((answer, guesses_sorted[i], answers, guesses) for answer in answers)
        )

        n_plays[:, i] = np.array([result[1] for result in results])

    mean_plays = np.nanmean(n_plays, axis=0)

    stop_time = time.time()

    mean_plays_sort_idx = np.argsort(mean_plays)

    print(
        f"Optimal start word: {guesses_sorted[mean_plays_sort_idx[0]]}, computed in {stop_time - start_time:.2f} seconds."
    )

    f = open("first_guess_deep_hard.csv", "w")
    f.write("guess,mean_plays\n")
    for i in range(n_guesses):
        f.write(
            f"{guesses_sorted[mean_plays_sort_idx[i]]},{mean_plays[mean_plays_sort_idx[i]]}\n"
        )
    f.close()
