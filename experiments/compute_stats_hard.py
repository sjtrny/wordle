from wordle import Game, HardAgent
import numpy as np
from multiprocessing import Pool, cpu_count
import time

def job(answer, answers, guesses):
    g = Game(word=answer, verbose=False)
    agent = HardAgent(answers, guesses, first_guess="reast")
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

    pool = Pool(cpu_count())
    start = time.time()
    results = pool.starmap(job, ((answer, answers, guesses) for answer in answers))
    end = time.time()

    n_plays = np.array([result[1] for result in results])

    failed_idx = np.arange(len(answers))[np.isnan(n_plays)]
    failed_words = [answers[idx] for idx in failed_idx]

    print(f"Mean Number of Plays: {np.nanmean(n_plays):.4f}")
    print(f"Number of failed words: {len(failed_words)}")
    print(f"Failed words: {', '.join(failed_words)}")
    print(f"Completed in words: {end-start:.4f} seconds")
