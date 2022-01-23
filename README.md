# wordle

A Python toolkit for solving wordle. The toolkit provides:
- an interactive solver
- game and agent classes to simulate playing a game

# Examples

Interactive solver. Run `demo_solver.py`

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

Simulate a game

    g = Game(word="crimp", verbose=True)
    agent = Agent(answers, guesses)
    final_guess, _ = agent.play(g)
    print(final_guess)

# Performance

The solver currently finishes in an average of 3.7099 plays.

The solver fails for the following 9 words: hatch, jaunt, store, watch, found, waste, jolly, dilly, gaunt.

Run `compute_stats.py` to replicate results.

# Methodology

The solver attempts to play a word that reduces the remaining possibilities as much as possible in each round. This is equivalent to playing the word that has the highest entropy for the outcome.

When guessing there are 243 possible outcomes (5 letters with 3 states i.e. 3^5). The entropy for a guess is thus 

![equation](http://www.sciweavers.org/tex2img.php?eq=-%20%5Csum_%7Bi%3D1%7D%5E%7B243%7D%20P%28o_i%29%20%5Clog%7BP%28o_i%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where P(o_i) is the probability of outcome i. The value of P(o_i) is just the proportion
of answers that fall in outcome i when playing the guess word.

The word that most evenly divides the answer pool into the 243 bins (i.e. highest entropy) will neccesarily result in a large number of small bins and thus the fewest options once played.

