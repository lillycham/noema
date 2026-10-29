# noema

Small transformers trained on synthetic worlds, to see what structure they learn.

The question: if a model sees worlds of varying dimension (and kind), does it learn what they have
in common, so that it can handle a world it never saw? The plan has four steps, each useful alone:

1. **1D cellular automata** as raw tokens. The model is never told the rule. It works it out from the rows.
2. A **2D world**, flattened to a sequence.
3. **Trajectories** in a simulated world, as raw coordinates, probed for what they encode.
4. **Mixed dimensions**, with one dimension held out.

Steps 1 and 2 are built. Nothing here is language: the tokens are `0`, `1` and a separator.

## Layout

| File                 | What it does                                                                  |
|----------------------|-------------------------------------------------------------------------------|
| `noema/ca.py`        | The 1D automaton. `step()` is the fast version, `reference_step()` the slow check |
| `noema/ca2d.py`      | The 2D automaton on a torus, and its eight symmetries                         |
| `noema/tokens.py`    | Rollout ↔ tokens, and which targets count in the loss                         |
| `noema/data.py`      | Random rule and first row, rolled out. Whole classes of rules held out        |
| `noema/data2d.py`    | The same for the 2D world: a frame is written as one long row                 |
| `noema/gpt.py`       | The model: RMSNorm, RoPE, QK-norm, SwiGLU, no biases                          |
| `noema/evaluate.py`  | Loss, accuracy row by row, and a rollout the model writes itself              |
| `noema/free_running.py` | A model that writes alone, against an ideal learner that does the same      |
| `noema/circuit.py`   | Which heads gather the neighbourhood and which look up the rule               |
| `noema/render.py`    | Rollouts as text and as pictures                                              |
| `noema/train_eca.py` | The training script for 1D                                                    |
| `noema/train_2d.py`  | The training script for 2D                                                    |
| `tests/`             | Tiny models on the CPU, a few seconds. `test_ca.py` and `test_ca2d.py` check `step()` against the slow version |

## Usage

```sh
nix develop                                    # or: direnv allow
pytest

python -m noema.train_eca --steps 3000 --tag first
python -m noema.train_2d --steps 3000 --tag first      # the 2D world
```

Pass `--slow-ca` to `train_eca` to use the slow reference step instead.

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

## The 2D world

`noema.train_2d` trains on a 2D automaton on a torus. A cell looks at itself and its four neighbours,
so a rule is again 32 bits (`noema/ca2d.py`). A frame of H×W cells is written line by line as one long
row with one separator. So the model, the loss and the scoring are the same as in 1D, and the grid
shows only through position: the cell above is W tokens back, and the same cell in the frame before
is H×W + 1 back.

The default grid is 5×5 with 8 frames, 208 tokens. A bigger grid gives the rule away at once: on 8×8
the ideal learner is at 0.78 after one frame, and on 5×5 it climbs from 0.64 to 1.00. Rules come in
classes of up to 16 (the eight turns and reflections of a square, and swapping 0 and 1), and
`data2d.RuleSet` holds out whole classes, 1 in 16.

On the first run (6,000 steps, two seeds each) three models reached the ideal learner's accuracy of
0.912. With RoPE that took about 1,000 steps. With learned positions it took about 2,500 steps, and
one seed stalled at 0.868.

## Looking inside

`python -m noema.circuit checkpoints/NAME.pt` looks for the circuit that works out the rule. It
reads from the checkpoint whether the world is 1D or 2D. For each head it prints the share of
attention that lands on earlier cells with the same transition as the one being predicted, the
distances back at which the head looks, and the accuracy with that head zeroed. `--by-cell
LAYER:HEAD:DISTANCE` draws the attention of one head at one distance for each cell it predicts.

In every model there are two stages. Gather heads fetch the cells of the neighbourhood from the row
or frame before, each head at one fixed distance. Retrieval heads then attend to the earlier cell with
the same neighbourhood number and copy what followed it. `noema.free_running` holds a model that
writes alone against an ideal learner that does the same.

## Renting a GPU

The Mac does about 1.3 s a step. A 5090 does about 20 ms, and the data is made as the model trains,
so there is nothing to upload. Find a machine with
`nix run nixpkgs#vastai -- search offers 'gpu_name=RTX_5090 num_gpus=1 reliability>0.98' -o dph`
(add `--raw` to see the price). A 5090 needs CUDA 12.8 or newer: `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime`
worked, with a 20 GB disk.

Clone the repo on the box and run `EXTRA="--radius 2" scripts/rent_eca.sh` (see its header). It installs
gcc if it can, because `torch.compile` needs a compiler and that image has none. Then it checks the
tests, times a short smoke run, and runs RoPE against learned positions on two seeds. The model is so
small that one GPU can run several of these at once. For the 2D world run
`TRAIN=noema.train_2d PREFIX=world2d STEPS=6000 scripts/rent_eca.sh`.

Destroy the instance as soon as the results are copied back. The command asks y/N, so in a script:
`echo y | nix run nixpkgs#vastai -- destroy instance ID`. The first radius-2 run took 17 minutes and cost
about $0.12. The first 2D run took 7 minutes of training and cost $0.06.

## Notes

The plan, the reasons and the related work are in the research notes (`research/dimension-worlds`).
