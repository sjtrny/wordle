import time
import requests
import pandas as pd

with open("../words_answers.txt", "r") as answers_file:
    answers = answers_file.read().splitlines()
with open("../words_guesses.txt", "r") as guesses_file:
    guesses = guesses_file.read().splitlines()

guess_data_list = []

for idx, guess in enumerate(answers):
    print(idx,guess)
    time.sleep(1)
    while True:
        try:
            resp = requests.get(url=f"https://books.google.com/ngrams/json?content={guess}&year_start=1800&year_end=2019&corpus=26&smoothing=0")
            data = resp.json() # Check the JSON Response Content documentation below
            guess_data_list.append(data[0])
            break
        except:
            time.sleep(10)


pd.DataFrame(guess_data_list).to_csv("ngram_data.csv", index=False)

# print(data)