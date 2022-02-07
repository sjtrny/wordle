from wordle import Game, MaxInfoAgent, MaxSplitsAgent, MaxPruneAgent
from multiprocessing import Pool, cpu_count
import time
import pandas as pd


def job(job_dict, answers, guesses):

    g = Game(word="DUMMY_WORD", max_plays=1, verbose=False)

    start_time = time.time()

    agent = job_dict["agent"](answers, guesses, first_guess=None)
    final_guess, _ = agent.play(g)

    end_time = time.time()

    job_dict.update(
        {
            "agent_name": job_dict["agent"].__name__,
            "first_word": final_guess,
            "time_taken": end_time - start_time,
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

    pd.DataFrame(results).to_csv("first_guess_shallow.csv", index=False)
