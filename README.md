# wordle

A fast solver for wordle written in Python using numba.

On my MacBook Pro (13-inch, M1, 2020) it takes:
- 3 seconds to compute the optimal word for the first play
- 30 minutes to compute the optimal word considering all outcomes

## Interactive Solver

Run `demos/demo_solver.py`

    Optimal starting guess: reast
    -----------------
    Enter guess,code: reast,10000
    Next guess: courd
    -----------------
    Enter guess,code: courd,20010
    Next guess: crick
    -----------------
    Enter guess,code: crick,22200
    Answer is crimp

## Simulate a game

Run `demos/demo_game.py`

    g = Game(word="crimp", verbose=True)
    agent = StandardAgent(answers, guesses)
    final_guess, _ = agent.play(g)
    print(final_guess)

## Game Modes

Wordle can be played in standard or hard mode. In hard mode, any revealed hints must be used in subsequent guesses.

This package supports both standard and hard mode. For simulations you can choose between StandardAgent or HardAgent, while
for interactive solving you can choose between StandardSolver and HardSolver.

## Optimal First Word

In terms of fewest average plays the optimal first word is `reast` when
considering all possible outcomes.

When considering only the first play the optimal first word is `soares`.

Run either `first_guess_{deep,shallow}_{standard,hard}.py` to replicate results.

## Performance

### Standard Mode

- Average number of plays to solve: 3.6004
- Failed words: None

### Hard Mode

- Average number of plays to solve: 3.6396
- Failed words: 9 total - `goner, hatch, jaunt, found, waste, taunt, catch, dilly, boxer`

### Notes

Failed words are often due to "lookalikes". For example with the word `hatch` the solver will check `match`, `batch`, `patch` and `latch` first and ultimately fail.

## Methodology

The solver attempts to play words that maximise the amount of information about the solution that is received after playing.
In statistical terms this means playing the word that has the highest entropy for the outcome.

A guess results in 243 possible outcomes (5 letters with 3 states i.e. 3^5). For example all 5 grey letters is one outcome, a green followed by four grey letters is another.

The entropy for a guess is given by 

![equation](http://www.sciweavers.org/tex2img.php?eq=-%20%5Csum_%7Bi%3D1%7D%5E%7B243%7D%20P%28o_i%29%20%5Clog%7BP%28o_i%29%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where P(o_i) is the probability of outcome i. The value of P(o_i) is just the proportion
of answers that fall in outcome i when playing the guess word.

The word that most evenly divides the answer pool into the 243 bins (i.e. highest entropy) will neccesarily result in a large number of small bins. Once the outcome is observed we are left to choose amongst the relatively few remaining answers associated with the bin.

## Installing numba on Apple M1

numba requires llvmlite, which in turn requires llvm version 11. The default installed version of llvm is likely more recent than version 11.

1. Install llvm version 11

`arch -arm64 brew install llvm@11`

2. Install llvmlite by pointing to old llvm version

`LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_3/bin/llvm-config" arch -arm64 pip install llvmlite`

`pip install numba`

## Installing scipy on Apple M1

    brew install openblas
    pip install --no-cache --no-use-pep517 pythran cython pybind11 gast
    OPENBLAS="$(brew --prefix openblas)" pip install --no-cache --no-binary :all: --no-use-pep517 scipy



