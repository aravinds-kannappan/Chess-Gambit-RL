# Shannon's Gambit: the theory, and what we actually found

## Motivation

Claude Shannon bookends this project. His 1948 paper founded information theory;
his 1950 paper founded computer chess. Shannon's Gambit treats chess through both
lenses at once: as a decision process to be optimised, and as a source of
uncertainty to be quantified in bits.

The aim is not to build the strongest engine. It is to connect three bodies of
theory - information theory, the Bellman/MDP formalism, and deep reinforcement
learning - into one honest, reproducible system trained on real human games, and
to be candid about where each method works and where it does not.

## 1. Chess as a Markov Decision Process

A position is a state; legal moves are actions; the opponent's reply is part of
the environment dynamics; the reward is +1 for delivering mate, 0 for a draw,
with a discount factor `gamma` that makes faster mates worth more. Because both
sides move, the correct operator is the **game (minimax) Bellman operator**:

```
V(s) = max_a  gamma * V(s')      if it is the maximiser's move
V(s) = min_a  gamma * V(s')      if it is the minimiser's move
```

Full chess has on the order of 10^44 states, far too many to enumerate. So we
make the Bellman pillar *provably correct* on a tractable but real subspace: the
king-and-rook (KRvK) and king-and-queen (KQvK) endgames.

### Result (exact)

KRvK enumerates **399,112** legal states. Value iteration - implemented as a
couple of vectorised NumPy segment-reductions per sweep - converges in **~33
sweeps** to a residual of `0`. The derived greedy policy **forces mate from 100%
of won positions**, and the value encodes distance-to-mate exactly (`V = gamma^d`),
matching the number of plies the optimal policy actually takes. This is the
verification that "Bellman works": not a loss curve, but a provable guarantee.

## 2. Learning the optimum from experience

Tabular Q-learning never sees the value table. It learns from sampled Bellman
backups, `Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]`. The
challenge is sparse reward over a huge state space: from random won positions the
agent almost never stumbles into mate. A **mate-distance curriculum** (start near
mate, expand outward - itself read off the exact value table) fixes this.

### Result (learned)

With the curriculum, tabular Q converts roughly **70% of positions within 4-8
plies of mate** against a random defender, degrading gracefully with distance.
That is genuine learning from experience, recovering much of the exact optimum
without ever being told it.

## 3. Deep reinforcement learning

Two deep-RL methods sit in the stack:

**DQN.** A convolutional Q-network over the 18-plane board encoding, with double
-DQN, a target network, and replay. Trained on the endgame MDP where we hold exact
ground truth.

**AlphaZero-lite.** A residual policy/value network with PUCT MCTS and self-play,
bootstrapped from supervised behavioural cloning so a laptop run refines a
sensible network rather than starting from noise.

### An honest limitation

A flat 4672-action DQN must learn a *position-specific mapping* from board to the
single correct mating move. That generalises poorly from sparse reward: observed
greedy conversion barely exceeds a random baseline (e.g. ~0.14 vs ~0.10 for
mate-in-2), and tabular memorisation comfortably beats it. We report this rather
than hide it. To show how much the Bellman solution helps, we feed the exact
value `V*` back into DQN as a **potential-based shaping reward** - which, by the
shaping theorem, leaves the optimal policy unchanged while densifying the signal.
The broader lesson is the historically correct one: for chess, **policy/value
networks with search (AlphaZero) are the effective deep-RL approach**, not flat
-action value learning.

## 4. The information-theoretic lens

Trained on real games, several measures become meaningful:

- **Move-choice entropy** `H(move | position)` measures how forced a position is.
- **KL / JS divergence** between policies (agent vs human vs engine) measures
  alignment.
- **Mutual information** `I(feature; result)` ranks which board features actually
  carry information about who wins. Empirically, **material difference carries the
  most**, ahead of mobility - quantitatively confirming chess intuition in bits.
- **Information gain per ply**, the drop in outcome entropy `H(result | position)`
  across a game, pinpoints *where the game was decided*: entropy stays high while
  the result is in doubt and collapses at the critical moment.

These are not decorations; entropy regularisation is also available in the policy
objective, so the same theory that measures the agent can shape it.

## 5. The prediction model

A single residual network with four heads - next-move (policy), value, win/draw
/loss, and player rating - is trained jointly on real games. The WDL and rating
heads power the live-prediction page; the policy head provides the MCTS prior and
the bootstrap for self-play. It is released on the Hugging Face Hub under
Apache-2.0 with a model card.

## What to take away

Exact methods and learned methods are not rivals here; they are a ladder. Bellman
value iteration gives ground truth on a solvable slice of chess; tabular Q shows
that ground truth is learnable from experience; the exact value even bootstraps
deep RL through reward shaping; and where tabular methods cannot scale, policy
/value networks with search take over. Information theory runs alongside the whole
ladder, turning "this position is sharp" or "this move was decisive" into bits.
And the one place a method underperforms - flat-action DQN - is reported plainly,
because a reproducible project should show the failures as clearly as the wins.
