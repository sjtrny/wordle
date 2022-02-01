import numpy as np
import string
from numba import jit
from abc import ABC, abstractmethod
import array

alphabet = string.ascii_lowercase


def get_match_code_game(guess, answer):
    result = array.array("u", ["0", "0", "0", "0", "0"])

    answer_char_count = np.zeros((len(alphabet)), dtype=int)

    for letter in answer:
        answer_char_count[ord(letter) - 97] += 1

    # Find exact matches
    for idx, letter in enumerate(guess):
        if letter == answer[idx]:
            result[idx] = "2"
            alphabet_idx = ord(letter) - 97
            answer_char_count[alphabet_idx] -= 1

    # Find inexact matches
    for idx, letter in enumerate(guess):
        alphabet_idx = ord(letter) - 97
        if answer_char_count[alphabet_idx] > 0 and letter != answer[idx]:
            result[idx] = "1"
            answer_char_count[alphabet_idx] -= 1

    return result.tounicode()


@jit(nopython=True, nogil=True, cache=True)
def get_match_code_int(guess_numba, answer_numba, answer_char_counts):
    result = 0
    answer_char_counts = answer_char_counts.copy()

    # Find exact matches
    for letter_idx in range(5):
        alphabet_idx = guess_numba[letter_idx]

        if guess_numba[letter_idx] == answer_numba[letter_idx]:
            result += 2 << letter_idx * 2
            answer_char_counts[alphabet_idx] -= 1

    # Find inexact matches
    for letter_idx in range(5):
        alphabet_idx = guess_numba[letter_idx]

        # If letter in answer
        if (
            answer_char_counts[alphabet_idx] > 0
            and guess_numba[letter_idx] != answer_numba[letter_idx]
        ):
            result += 1 << letter_idx * 2
            answer_char_counts[alphabet_idx] -= 1

    return result


@jit(nopython=True, nogil=True, cache=True)
def get_bin_counts(guesses, answers, answers_char_counts):
    n_codes = 1024  # 2 bits for 5 characters gives max 1024

    counts = np.zeros((n_codes, guesses.shape[0]), dtype=np.intc)

    for guess_idx in range(guesses.shape[0]):

        guess_array = guesses[guess_idx, :]

        for answer_idx in range(answers.shape[0]):
            result = get_match_code_int(
                guess_array,
                answers[answer_idx, :],
                answers_char_counts[answer_idx, :],
            )
            counts[result, guess_idx] += 1

    return counts


# Using Dvoretzky-Kiefer-Wolfowitz inequality
def min_samples(eps, alpha):
    return (1 / (2 * eps**2)) * np.log(2 / alpha)


@jit(nopython=True, nogil=True, cache=True)
def get_bin_counts_approximate(guesses, answers, answers_char_counts, sample_size):
    n_codes = 1024  # 2 bits for 5 characters gives max 1024

    counts = np.zeros((n_codes, guesses.shape[0]), dtype=np.intc)

    sample_size = (np.minimum(len(answers), sample_size),)

    for guess_idx in range(guesses.shape[0]):
        guess_array = guesses[guess_idx, :]

        sub_answer_idxs = np.random.choice(len(answers), size=sample_size, replace=True)

        for sub_answer_idx in range(sub_answer_idxs.shape[0]):
            answer_idx = sub_answer_idxs[sub_answer_idx]

            result = get_match_code_int(
                guess_array,
                answers[answer_idx, :],
                answers_char_counts[answer_idx, :],
            )
            counts[result, guess_idx] += 1

    return counts


@jit(nopython=True, nogil=True, cache=True)
def get_bin_table(guesses, answers, answers_char_counts):
    bin_table = np.zeros((guesses.shape[0], answers.shape[0]), dtype=np.uint)

    for guess_idx in range(guesses.shape[0]):
        guess_array = guesses[guess_idx, :]

        for answer_idx in range(answers.shape[0]):
            bin_table[guess_idx, answer_idx] = get_match_code_int(
                guess_array,
                answers[answer_idx, :],
                answers_char_counts[answer_idx, :],
            )

    return bin_table.T


@jit(nopython=True, nogil=True, cache=True)
def bin_table_to_counts(bin_table, guess_mask, answer_mask):
    n_codes = 1024  # 2 bits for 5 characters gives max 1024

    n_guesses = np.sum(guess_mask)

    counts = np.zeros((n_codes, n_guesses), dtype=np.intc)

    # Returns non zero index for each axis, get first axis
    guess_idxs = np.nonzero(guess_mask)[0]
    answer_idxs = np.nonzero(answer_mask)[0]

    for guess_idx in range(len(guess_idxs)):
        for answer_idx in answer_idxs:
            counts[bin_table[answer_idx, guess_idxs[guess_idx]], guess_idx] += 1
    return counts


@jit(nopython=True, nogil=True, cache=True)
def mask_candidates(match_int, guess_numba, words_numba, word_char_counts, mask):
    match_codes = np.full((mask.shape[0],), 0)

    for idx in range(mask.shape[0]):
        if mask[idx]:
            match_codes[idx] = get_match_code_int(
                guess_numba, words_numba[idx, :], word_char_counts[idx, :]
            )

    return mask & (match_codes == match_int)


def entropy(counts):
    probabilities = counts / np.sum(counts, axis=0)

    return -np.sum(
        probabilities
        * np.log(
            probabilities, out=np.zeros_like(probabilities), where=probabilities != 0
        ),
        axis=0,
    )


def get_numeric_representations(wordlist):
    words_numba = np.zeros((len(wordlist), 5), dtype=np.intc)
    words_char_counts = np.zeros((len(wordlist), len(alphabet)), dtype=np.intc)
    for w, word in enumerate(wordlist):
        for letter_idx, letter in enumerate(word):
            words_numba[w, letter_idx] = ord(letter) - 97
            words_char_counts[w, ord(letter) - 97] += 1

    return words_numba, words_char_counts


class Game:
    def __init__(self, word=None, answers=None, verbose=False):
        if word:
            self.word = word
        elif answers:
            self.word = np.random.choice(answers)
        else:
            raise ValueError("Must provide word or answers.")

        self.max_plays = 6
        self.guess_list = []
        self.verbose = verbose

    def play(self, guess):
        if len(self.guess_list) < self.max_plays - 1:
            if guess != self.word:
                self.guess_list.append(guess)
                code = get_match_code_game(guess, self.word)
                if self.verbose:
                    print(guess, code)
                return code

        return None


class Agent(ABC):
    @abstractmethod
    def __init__(self, answers, guesses, mode="standard", first_guess=None):
        pass

    @abstractmethod
    def play(self, game):
        pass


class NumbaAgent(Agent):
    def __init__(self, answers, guesses, mode="standard", first_guess=None):
        self.answers = answers
        self.guesses = guesses

        self.guesses_numba, self.guesses_char_counts = get_numeric_representations(
            self.guesses
        )
        self.answers_numba, self.answers_char_counts = get_numeric_representations(
            self.answers
        )

        self.mode = mode
        self.first_guess = first_guess

    @abstractmethod
    def play(self, game):
        pass


class MaxInfoAgent(NumbaAgent):
    def __init__(self, answers, guesses, mode="standard", first_guess="reast"):
        super().__init__(answers, guesses, mode, first_guess)

    def play(self, game):

        guess_history = []
        code_history = []

        guess_history.append(self.first_guess)
        state = game.play(self.first_guess)
        code_history.append(state)

        guess_total_mask = np.ones(len(self.guesses)).astype(bool)
        answer_total_mask = np.ones(len(self.answers)).astype(bool)

        while state:
            guess_idx = self.guesses.index(guess_history[-1])
            # Translate match_code into integer
            match_int = 0
            for idx, c in enumerate(code_history[-1]):
                match_int += int(c) << idx * 2

            if self.mode == "standard":
                # Exclude previously guessed words
                guess_total_mask[guess_idx] = False
            else:
                guess_total_mask = mask_candidates(
                    match_int,
                    self.guesses_numba[guess_idx, :],
                    self.guesses_numba,
                    self.guesses_char_counts,
                    guess_total_mask,
                )

            answer_total_mask = mask_candidates(
                match_int,
                self.guesses_numba[guess_idx, :],
                self.answers_numba,
                self.answers_char_counts,
                answer_total_mask,
            )

            if np.sum(answer_total_mask) > 1:
                # For whatever reason, indexing like this adds a dimension so we
                # squeeze the dimensions

                bin_counts = get_bin_counts(
                    self.guesses_numba[guess_total_mask, :].squeeze(),
                    self.answers_numba[answer_total_mask, :].squeeze(),
                    self.answers_char_counts[answer_total_mask, :].squeeze(),
                )

                # Compute Entropy
                guesses_entropy = entropy(bin_counts)

                # Sort entropy
                sort_idx = np.argsort(guesses_entropy)

                # Select best word
                remaining_guesses = np.array(self.guesses)[guess_total_mask]
                remaining_guesses_sorted = remaining_guesses[sort_idx]

                guess = remaining_guesses_sorted[-1]

            else:
                guess = np.array(self.answers)[answer_total_mask][0]

            guess_history.append(guess)

            state = game.play(guess)
            code_history.append(state)

        return guess_history[-1], len(guess_history)


class MaxSplitsAgent(NumbaAgent):
    def __init__(self, answers, guesses, mode="standard", first_guess="reast"):
        super().__init__(answers, guesses, mode, first_guess)

    def play(self, game):

        guess_history = []
        code_history = []

        guess_history.append(self.first_guess)
        state = game.play(self.first_guess)
        code_history.append(state)

        guess_total_mask = np.ones(len(self.guesses)).astype(bool)
        answer_total_mask = np.ones(len(self.answers)).astype(bool)

        while state:
            guess_idx = self.guesses.index(guess_history[-1])
            # Translate match_code into integer
            match_int = 0
            for idx, c in enumerate(code_history[-1]):
                match_int += int(c) << idx * 2

            guess_total_mask = mask_candidates(
                match_int,
                self.guesses_numba[guess_idx, :],
                self.guesses_numba,
                self.guesses_char_counts,
                guess_total_mask,
            )
            answer_total_mask = mask_candidates(
                match_int,
                self.guesses_numba[guess_idx, :],
                self.answers_numba,
                self.answers_char_counts,
                answer_total_mask,
            )

            if np.sum(answer_total_mask) > 1:
                # For whatever reason, indexing like this adds a dimension so we
                # squeeze the dimensions
                bin_counts = get_bin_counts(
                    self.guesses_numba[guess_total_mask, :].squeeze(),
                    self.answers_numba[answer_total_mask, :].squeeze(),
                    self.answers_char_counts[answer_total_mask, :].squeeze(),
                )

                # Compute Number of Splits
                guesses_nsplits = np.count_nonzero(bin_counts, axis=0)

                # Sort splits
                sort_idx = np.argsort(guesses_nsplits)

                # Select best word
                remaining_guesses = np.array(self.guesses)[guess_total_mask]
                remaining_guesses_sorted = remaining_guesses[sort_idx]

                guess = remaining_guesses_sorted[-1]

            else:
                guess = np.array(self.answers)[answer_total_mask][0]

            guess_history.append(guess)

            state = game.play(guess)
            code_history.append(state)

        return guess_history[-1], len(guess_history)


class MaxPruneAgent(NumbaAgent):
    def __init__(self, answers, guesses, mode="standard", first_guess="reast"):
        super().__init__(answers, guesses, mode, first_guess)

    def play(self, game):

        guess_history = []
        code_history = []

        guess_history.append(self.first_guess)
        state = game.play(self.first_guess)
        code_history.append(state)

        guess_total_mask = np.ones(len(self.guesses)).astype(bool)
        answer_total_mask = np.ones(len(self.answers)).astype(bool)

        while state:
            guess_idx = self.guesses.index(guess_history[-1])
            # Translate match_code into integer
            match_int = 0
            for idx, c in enumerate(code_history[-1]):
                match_int += int(c) << idx * 2

            guess_total_mask = mask_candidates(
                match_int,
                self.guesses_numba[guess_idx, :],
                self.guesses_numba,
                self.guesses_char_counts,
                guess_total_mask,
            )
            answer_total_mask = mask_candidates(
                match_int,
                self.guesses_numba[guess_idx, :],
                self.answers_numba,
                self.answers_char_counts,
                answer_total_mask,
            )

            if np.sum(answer_total_mask) > 1:

                filtered_guesses_numba = self.guesses_numba[
                    guess_total_mask, :
                ].squeeze()
                filtered_answers_numba = self.answers_numba[
                    answer_total_mask, :
                ].squeeze()
                filtered_answers_char_counts = self.answers_char_counts[
                    answer_total_mask, :
                ].squeeze()

                total_unmatched = np.zeros(
                    filtered_guesses_numba.shape[0], dtype=np.intc
                )

                for guess_idx in range(filtered_guesses_numba.shape[0]):

                    guess_array = filtered_guesses_numba[guess_idx, :]

                    for answer_idx in range(filtered_answers_numba.shape[0]):
                        match_int = get_match_code_int(
                            guess_array,
                            filtered_answers_numba[answer_idx, :],
                            filtered_answers_char_counts[answer_idx, :],
                        )

                        if match_int == 0:
                            total_unmatched[guess_idx] += 1

                # Sort splits
                sort_idx = np.argsort(total_unmatched)

                # Select best word
                remaining_guesses = np.array(self.guesses)[guess_total_mask]
                remaining_guesses_sorted = remaining_guesses[sort_idx]

                guess = remaining_guesses_sorted[0]

            else:
                guess = np.array(self.answers)[answer_total_mask][0]

            guess_history.append(guess)

            state = game.play(guess)
            code_history.append(state)

        return guess_history[-1], len(guess_history)


class Solver(ABC):
    @abstractmethod
    def __init__(self, answers, guesses, mode="standard"):
        pass

    @abstractmethod
    def step(self, code=None, guess=None):
        pass


class NumbaSolver(Solver):
    def __init__(self, answers, guesses, mode="standard"):
        self.answers = answers
        self.guesses = guesses

        self.guesses_numba, self.guesses_char_counts = get_numeric_representations(
            self.guesses
        )
        self.answers_numba, self.answers_char_counts = get_numeric_representations(
            self.answers
        )

        self.guess_history = []
        self.code_history = []

        self.guess_total_mask = np.ones(len(self.guesses)).astype(bool)
        self.answer_total_mask = np.ones(len(self.answers)).astype(bool)

        self.mode = mode


class MaxInfoSolver(NumbaSolver):
    def step(self, code=None, guess=None):
        if code:
            self.guess_history.append(guess)
            self.code_history.append(code)

            guess_idx = self.guesses.index(self.guess_history[-1])

            # Translate match_code into integer
            match_int = 0
            for idx, c in enumerate(self.code_history[-1]):
                match_int += int(c) << idx * 2

            # Exclude previously guessed words
            if self.mode == "standard":
                # Exclude previously guessed words
                self.guess_total_mask[guess_idx] = False
            else:
                self.guess_total_mask = mask_candidates(
                    match_int,
                    self.guesses_numba[guess_idx, :],
                    self.guesses_numba,
                    self.guesses_char_counts,
                    self.guess_total_mask,
                )

            self.answer_total_mask = mask_candidates(
                match_int,
                self.guesses_numba[guess_idx, :],
                self.answers_numba,
                self.answers_char_counts,
                self.answer_total_mask,
            )

        if np.sum(self.answer_total_mask) > 1:
            # For whatever reason, indexing like this adds a dimension so we
            # squeeze the dimensions
            bin_counts = get_bin_counts(
                self.guesses_numba[self.guess_total_mask, :].squeeze(),
                self.answers_numba[self.answer_total_mask, :].squeeze(),
                self.answers_char_counts[self.answer_total_mask, :].squeeze(),
            )

            # Compute Entropy
            guesses_entropy = entropy(bin_counts)

            # Sort by entropy
            sort_idx = np.argsort(guesses_entropy)

            remaining_guesses = np.array(self.guesses)[self.guess_total_mask]
            remaining_guesses_sorted = remaining_guesses[sort_idx]

            guess = remaining_guesses_sorted[-1]

        else:
            guess = np.array(self.answers)[self.answer_total_mask][0]

        return guess, np.array(self.answers)[self.answer_total_mask]
