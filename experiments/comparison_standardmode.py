from wordle import Game, MaxInfoAgent, MaxSplitsAgent, MaxPruneAgent
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import pandas as pd


def job(job_dict, answers, guesses):
    results = []

    for answer in answers:
        g = Game(word=answer, verbose=False)
        start_time = time.time()
        agent = job_dict["agent"](answers, guesses, mode="standard")
        end_time = time.time()
        final_guess, n_guesses = agent.play(g)
        results.append((final_guess, n_guesses, end_time - start_time))

    n_plays = np.array([result[1] for result in results])
    mean_n_plays = np.nanmean(n_plays)
    failed_idx = np.arange(len(answers))[np.isnan(n_plays)]
    failed_words = [answers[idx] for idx in failed_idx]

    play_times = np.array([result[2] for result in results])
    play_times[play_times == 0] = np.NaN

    mean_play_time = np.nanmean(play_times)

    job_dict.update(
        {
            "agent_name": job_dict["agent"].__name__,
            "mean_n_plays": mean_n_plays,
            "failed_words": failed_words,
            "mean_play_time": mean_play_time,
        }
    )

    return job_dict


if __name__ == "__main__":
    with open("../words_answers.txt", "r") as answers_file:
        answers = answers_file.read().splitlines()
    with open("../words_guesses.txt", "r") as guesses_file:
        guesses = guesses_file.read().splitlines()

    agent_list = [MaxInfoAgent, MaxSplitsAgent, MaxPruneAgent]

    pool = Pool(cpu_count())

    results = pool.starmap(
        job, (({"agent": agent}, answers, guesses) for agent in agent_list)
    )

    pd.DataFrame(results).to_csv("comparison_standardmode.csv", index=False)
