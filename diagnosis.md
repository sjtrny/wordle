# Diagnosis

The goal of the game is to identify the true diagnosis from a set of diagnoses by sequentially making actions to reveal information about the diagnosis.

## Problem Statement

Given a set $\mathcal D = \{ d_i \}_{i=1}^D$ of $D$ of possible diagnoses, the goal of the game is to identify the true diagnosis, $d_*$.

The player starts with no information about the true diagnosis.

In earch turn the player plays an action (performs a test) $a_t$ from a set of possible actions $\mathcal A = \{ a_i \}_{i=1}^{A}$.

After each action feedback is given that reduces the set of possible diagnoses. In other words we start with $\mathcal D_t$, after the first play $\mathcal D_{t+1}$ diagnoses are remaining where $\mathcal D_{t+1} \subset \mathcal D_t$.

## Examples

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

