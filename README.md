# wordle

A very fast solver for wordle written in Python using numba.

# Interactive Solver

Run `demo_solver.py`

    Optimal starting guess: soare
    -----------------
    Enter guess,code: soare,00010
    Next guess: trild, 175 valid guess words remaining
    -----------------
    Enter guess,code: trild,02200
    Next guess: prink, 21 valid guess words remaining
    -----------------
    Enter guess,code: prink,12200
    Answer is crimp

# Simulate a game

Run `demo_game.py`

    g = Game(word="crimp", verbose=True)
    agent = Agent(answers, guesses)
    final_guess, _ = agent.play(g)
    print(final_guess)

# Optimal First Word

In terms of fewest average plays the optimal first word is `reast` when
considering all possible outcomes.

When considering only the first play the optimal first word is `soares`.

Run either `compute_first_word_{deep,shallow}.py` to replicate results.

# Performance

The solver currently finishes in an average of `3.6396` plays.

The solver fails for the following 9 words: `goner, hatch, jaunt, found, waste, taunt, catch, dilly, boxer`.

Run `compute_stats.py` to replicate results.

# Methodology

The solver attempts to play a word that reduces the remaining possibilities as much as possible in each round. This is equivalent to playing the word that has the highest entropy for the outcome.

When guessing there are 243 possible outcomes (5 letters with 3 states i.e. 3^5). The entropy for a guess is thus 

![equation](http://www.sciweavers.org/tex2img.php?eq=-%20%5Csum_%7Bi%3D1%7D%5E%7B243%7D%20P%28o_i%29%20%5Clog%7BP%28o_i%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where P(o_i) is the probability of outcome i. The value of P(o_i) is just the proportion
of answers that fall in outcome i when playing the guess word.

The word that most evenly divides the answer pool into the 243 bins (i.e. highest entropy) will neccesarily result in a large number of small bins. Once the outcome is observed we are left to choose amongst the relatively few remaining answers associated with the bin.

