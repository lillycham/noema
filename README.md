# noema

Small transformers trained on synthetic worlds, to see what structure they learn.

The question: if a model sees worlds of varying dimension (and kind), does it learn what they have
in common, so that it can handle a world it never saw? The plan has four steps, each useful alone:

1. **1D cellular automata** as raw tokens. The model is never told the rule. It works it out from the rows.
2. A **2D world**, flattened to a sequence.
3. **Trajectories** in a simulated world, as raw coordinates, probed for what they encode.
4. **Mixed dimensions**, with one dimension held out.

This is step 1. Nothing here is language: the tokens are `0`, `1` and a row separator.

## Layout

| File                 | What it does                                                                  |
|----------------------|-------------------------------------------------------------------------------|
| `noema/ca.py`        | The automaton. `step()` is yours to write; `reference_step()` is the slow check |
| `noema/tokens.py`    | Rollout ↔ tokens, and which targets count in the loss                         |
| `noema/data.py`      | Random rule and first row, rolled out. Whole classes of rules held out        |
| `noema/gpt.py`       | The model: RMSNorm, RoPE, QK-norm, SwiGLU, no biases                          |
| `noema/evaluate.py`  | Loss, accuracy row by row, and a rollout the model writes itself              |
| `noema/render.py`    | Rollouts as text and as pictures                                              |
| `noema/train_eca.py` | The training script                                                           |
| `tests/`             | Tiny models on the CPU, a few seconds. `test_ca.py` checks your `step()`      |

## Usage

```sh
nix develop                                    # or: direnv allow
pytest                                         # 4 tests wait for step() in noema/ca.py

python -m noema.train_eca --slow-ca --steps 3000 --tag first
python -m noema.train_eca --steps 3000         # once step() is written
```

A run prints the loss and accuracy on rules it trained on and on held-out ones, then the accuracy
for each row, then saves a picture of six rollouts to `runs/` (the truth, the model, and where they
differ). The first rows are given; the model writes the rest.

## Held-out rules

ECA rules come in classes with the same dynamics: reflect the world, swap 0 and 1, or both. Rule 110
is the same automaton as 124, 137 and 193. The 256 rules are 88 classes. A split by rule number
would put 110 in the test set and 124 in training, and test nothing. `data.split_rules` holds out
whole classes.

## Notes

The plan, the reasons and the related work are in the research notes (`research/dimension-worlds`).
