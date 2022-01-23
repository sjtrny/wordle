from wordle import Game, Agent
import numpy as np
from multiprocessing import Pool, cpu_count

with open("words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("words_guesses.txt", "r") as guesses_file:
    guesses = guesses_file.read().splitlines()

n_plays = np.zeros(len(answers))


def job(
    answer,
):
    g = Game(word=answer, verbose=False)
    agent = Agent(answers, guesses)
    (
        final_guess,
        n_guesses,
    ) = agent.play(g)
    if final_guess == answer:
        return answer, n_guesses
    else:
        return answer, np.nan


if __name__ == "__main__":
    pool = Pool(cpu_count())
    results = pool.map(job, answers)

    n_plays = np.array([result[1] for result in results])

    failed_idx = np.arange(len(answers))[np.isnan(n_plays)]
    failed_words = [answers[idx] for idx in failed_idx]

    print(f"Mean Number of Plays: {np.nanmean(n_plays):.4f}")
    print(f"Number of failed words: {len(failed_words)}")
    print(f"Failed words: {', '.join(failed_words)}")
