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

## Radius 2

`--radius 2` makes a cell look two places each way: a rule is 32 bits, so there are 2^32 of them,
and no model can have learned any of them. A held-out rule is then really a new rule. One transition
on 15 cells shows only about half of the 32 neighbourhood types, so the rule has to be collected
across several rows. `evaluate.ideal_accuracy_by_row` is the curve of an ideal learner that keeps a
table of every transition it has seen and guesses the rest. At radius 2 it starts at 0.59 and
reaches 1.00 by row 9, and the model is held against it. A run prints both.

## Renting a GPU

The Mac does about 1.3 s a step. A 5090 does about 20 ms, and the data is made as the model trains,
so there is nothing to upload. Find a machine with
`nix run nixpkgs#vastai -- search offers 'gpu_name=RTX_5090 num_gpus=1 reliability>0.98' -o dph`
(add `--raw` to see the price). A 5090 needs CUDA 12.8 or newer: `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime`
worked, with a 20 GB disk.

Clone the repo on the box and run `EXTRA="--radius 2" scripts/rent_eca.sh` (see its header). It installs
gcc if it can, because `torch.compile` needs a compiler and that image has none. Then it checks the
tests, times a short smoke run, and runs RoPE against learned positions on two seeds. The model is so
small that one GPU can run several of these at once.

Destroy the instance as soon as the results are copied back. The command asks y/N, so in a script:
`echo y | nix run nixpkgs#vastai -- destroy instance ID`. The first radius-2 run took 17 minutes and cost
about $0.12.

## Notes

The plan, the reasons and the related work are in the research notes (`research/dimension-worlds`).
