from wordle import Game, MaxInfoAgent, MaxSplitsAgent, MaxPruneAgent
from wordle import get_numeric_representations, get_bin_table, bin_table_to_counts
from multiprocessing import Pool, cpu_count
import time
import pandas as pd
import numpy as np
import itertools


def sub_job(agent_cls, sub_guesses, answers, guesses):

    guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
    answers_numba, answers_char_counts = get_numeric_representations(answers)

    bin_table = get_bin_table(guesses_numba, answers_numba, answers_char_counts)

    sub_guess_results = []

    for guess_idx, first_guess in enumerate(sub_guesses):
        print(first_guess)
        for answer_idx, answer in enumerate(answers):
            g = Game(word=answer, verbose=False)

            start_time = time.time()
            agent = agent_cls(
                answers,
                guesses,
                mode="hard",
                first_guess=first_guess,
                guess_data=(guesses_numba, guesses_char_counts),
                answer_data=(answers_numba, answers_char_counts),
                bin_table=bin_table,
            )
            final_guess, n_guesses = agent.play(g)
            end_time = time.time()
            # print(end_time - start_time)

            sub_guess_results.append(
                (first_guess, answer, final_guess, n_guesses, end_time - start_time)
            )

    return sub_guess_results


def job(job_dict, answers, guesses):

    agent_cls = job_dict["agent"]
    print(agent_cls.__name__)

    top_n = np.minimum(len(guesses), job_dict["top_n"])

    # SELECT TOP N STARTING WORDS
    # This needs to change depending on the agent

    guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
    answers_numba, answers_char_counts = get_numeric_representations(answers)

    bin_table = get_bin_table(guesses_numba, answers_numba, answers_char_counts)

    bin_counts = bin_table_to_counts(
        bin_table, np.full(len(guesses), True), np.full(len(answers), True)
    )

    sort_idx = agent_cls.order_guesses(bin_counts)

    # Select best word
    guesses_sorted = np.array(guesses)[sort_idx]
    guesses_sorted = guesses_sorted[:top_n]

    split_guesses = np.array_split(guesses_sorted, cpu_count())

    pool = Pool(cpu_count())

    total_result_lists = pool.starmap(
        sub_job,
        (
            (
                job_dict["agent"],
                sub_guesses,
                answers,
                guesses,
            )
            for sub_guesses in split_guesses
        ),
    )

    total_results = list(itertools.chain.from_iterable(total_result_lists))

    return total_results


if __name__ == "__main__":
    with open("../words_answers.txt", "r") as answers_file:
        answers = answers_file.read().splitlines()
    with open("../words_guesses.txt", "r") as guesses_file:
        guesses = guesses_file.read().splitlines()

    agent_list = [MaxInfoAgent, MaxSplitsAgent, MaxPruneAgent]

    start = time.time()
    results_per_agent = [
        job({"agent": agent, "top_n": 100}, answers, guesses) for agent in agent_list
    ]
    end = time.time()
    print("completed in", end - start)

    cols = ["first_guess", "answer", "final_guess", "n_guesses", "play_time"]

    dataframes = [
        pd.DataFrame(agent_result, columns=cols) for agent_result in results_per_agent
    ]

    for dataframe_idx, dataframe in enumerate(dataframes):
        dataframe["agent"] = agent_list[dataframe_idx].__name__

    results = pd.concat(dataframes, axis=0)

    results[results["answer"] != "final_guess"][["n_guesses", "play_time"]] = np.NaN

    results.to_csv("first_guess_deep_hardmode.csv", index=False)

    # Summarise results

    summary = results.groupby(["agent", "first_guess"]).mean().sort_values("n_guesses")
    summary.to_csv("first_guess_deep_hardmode_summary.csv")
