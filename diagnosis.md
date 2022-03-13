# Diagnosis

The goal of the game is to identify the true diagnosis from a set of diagnoses by sequentially making actions to reveal information about the diagnosis.

## Problem Statement

Given a set $\mathcal D = \{ d_i \}_{i=1}^D$ of $D$ of possible diagnoses, the goal of the game is to identify the true diagnosis, $d_*$.

The player starts with no information about the true diagnosis.

In earch turn the player plays an action (performs a test) $a_t$ from a set of possible actions $\mathcal A = \{ a_i \}_{i=1}^{A}$.

After each action feedback is given that reduces the set of possible diagnoses. In other words we start with $\mathcal D_t$, after the first play $\mathcal D_{t+1}$ diagnoses are remaining where $\mathcal D_{t+1} \subset \mathcal D_t$.

## Solutions

A diagnosis can be found by an exhaustive search, however exhaustive searches are often time or cost prohibitive.

In such situations where exhaustive searches are impractical one must resort to following a policy in order to find the diagnosis in some sense of optimality.

Definitions of optimality vary and are contended. Commonly used metrics are:

- smallest mean number of guesses
- fewest cumulative guesses
- least number of failed diagnoses

Noting that these metrics are aggregates and are therefore computed over the set of all possible diagnoses. 

## Research Questions

1. Is there a globally optimal policy?
2. Why or why not?
3. Can efficient policies be developed for all variants?
4. Can we devleop policies for huge problems?
4. What approximiations can be deployed and can guarantees be developed for their success?

## Extensions

### Random Feedback

In most games the feedback given is correct. However in real world situations tests can be innacurate or even incorrect. Therefore we need to consider that the feedback we get is random.

Some research has been undertaken for a probabilistic version of Mastermind: [Bayesian networks in Mastermind](http://staff.utia.cas.cz/vomlel/mastermind.pdf).

Requirements for repeated testing

- If highly unusual outcome from test is received which would normally be re-tested how do we incorporate this into the maximum information criterion?

### Prior-Knowledge



### Playing blind



### Action Costs

In most games all action have equal costs. However in real world situations there may be different costs associated with each action.

Examples:
- how long it takes to perform an action
- how much money it costs to perform an action
- quality of life costs e.g. invasive surgery or procedures

> Usually, when a patient is being investigated, only a small subset of all available tests is performed. Most selection methods for making this choice fail to account for the risk and cost of the test. By attempting to approximate a decision-analytic ideal, via the concept of quasi-utility, the authors developed the information-to-cost ratio and related measures, which balance the utility of the information gained against the price paid.

[Test Selection Measures, 1989](https://journals.sagepub.com/doi/pdf/10.1177/0272989X8900900208)

### Pareto Optimality

These extensions introduce new optimality criteria, which must be satisfied alongside original criteria. In these situations one might accept any of the Pareto optimal solutions.


## Examples

### Research

- [Research as a Stochastic Decision Process](https://cs.stanford.edu/~jsteinhardt/ResearchasaStochasticDecisionProcess.html)

### Medical

- Royal Flying Doctor Service, "Where does it hurt?" radio aid chart
  - [Royal Flying Doctor Service Twitter](https://twitter.com/royalflyingdoc/status/439293912943841280)
  - [TikTok Explainer](https://www.tiktok.com/@julianoshea/video/7066607895537847554)
  - [NLA Archives](https://nla.gov.au/nla.obj-133663850/view)

### Classic Games

- [LINGO](https://en.wikipedia.org/wiki/Lingo_(American_game_show))
  - https://www.youtube.com/watch?v=T_iC26tnDqM
- [Bulls and Cows](https://en.wikipedia.org/wiki/Bulls_and_Cows)
- [Jotto](https://en.wikipedia.org/wiki/Jotto)
- [Mastermind](https://en.wikipedia.org/wiki/Mastermind_(board_game))
- [Guess Who](https://en.wikipedia.org/wiki/Guess_Who%3F)

### Games Inspired by Wordle

- [Quordle](https://www.quordle.com/)
- [Worldle](https://worldle.teuteuf.fr)
- [Nerdle](https://nerdlegame.com)
- [Mathle](https://mathlegame.com)
- [Primel](https://converged.yt/primel/)
- [Airportle](https://airportle.glitch.me/)
- [Three Magic Words](https://www.threemagicwords.app/play)
- [Mathler](https://www.mathler.com/)
- [Shaple](https://swag.github.io/shaple/)

### Adversarial Variants

- [Evil Wordle](https://swag.github.io/evil-wordle/)
- [Absurdle](https://qntm.org/files/absurdle/absurdle.html)

### Game Solvers (Reverse Problem)

- [Akinator](https://en.akinator.com/)

