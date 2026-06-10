# Imitation Learning for MiniWoB Click Tasks

This repo trains a small behavioral-cloning policy for simple MiniWoB++ browser
tasks. The model sees a screenshot plus the task instruction and predicts where
to click.

The policy does not read the DOM. The DOM is only used by scripted experts while
collecting demonstrations.

## What It Does

1. `il.collect` runs scripted experts and saves successful click examples.
2. `il.train` trains a CNN policy from those examples.
3. `il.evaluate` runs the trained policy in MiniWoB and reports success rate.

Default tasks:

```text
click-test
click-test-2
click-button
click-link
```

## Setup

Install dependencies:

```bash
uv sync
```

MiniWoB uses Selenium and Chrome. On macOS:

```bash
brew install --cask google-chrome
```

## Quickstart

Collect demonstrations:

```bash
uv run python -m il.collect
```

Train:

```bash
uv run python -m il.train
```

Evaluate:

```bash
uv run python -m il.evaluate
```

The main checkpoint is written to:

```text
runs/bc/best.pt
```

## Common Commands

Collect fewer demos for a quick run:

```bash
uv run python -m il.collect --episodes-per-task 50
```

Collect demos for one task:

```bash
uv run python -m il.collect --tasks click-button --episodes-per-task 50
```

Train with a specific config:

```bash
uv run python -m il.train --config configs/default.yaml
```

Evaluate with a visible browser:

```bash
uv run python -m il.evaluate --watch
```

Evaluate with a random-click baseline:

```bash
uv run python -m il.evaluate --baseline
```

Evaluate one task:

```bash
uv run python -m il.evaluate --tasks click-button
```

## Configuration

Most settings live in:

```text
configs/default.yaml
```

Important fields:

```yaml
tasks:                  # MiniWoB tasks to collect/train/evaluate on
collect:
  episodes_per_task:    # demos per task
model:
  heatmap_size:         # click grid size; default 16 means 16x16 cells
train:
  batch_size:
  epochs:
  lr:
  checkpoint_metric:    # loss, exact, or within-1
  out_dir:              # default runs/bc
eval:
  checkpoint:           # default runs/bc/best.pt
```

## Training Metrics

Training prints lines like:

```text
epoch 9 | train_loss 0.5198 | val_loss 1.6523 | exact 0.675 | within-1 0.738
```

`exact` is the fraction of validation examples where the predicted click grid
cell exactly matches the expert cell.

`within-1` is the fraction where the predicted cell is at most one grid cell
away from the expert cell, including diagonals. With the default `16x16` grid,
one cell is about `10 px` wide and `13 px` tall in the MiniWoB canvas.

## Project Layout

```text
configs/default.yaml   default tasks and hyperparameters
src/il/env.py          MiniWoB environment wrapper
src/il/experts.py      scripted click experts
src/il/collect.py      demonstration collection
src/il/dataset.py      demo loading and pixel-to-grid conversion
src/il/model.py        screenshot + instruction click policy
src/il/train.py        behavioral cloning training loop
src/il/evaluate.py     live MiniWoB evaluation
data/demos/            collected demonstrations
runs/bc/               checkpoints
```

## Notes

This is intentionally narrow: it handles click-only tasks. Typing tasks or
multi-step browser tasks need a richer action space, such as an action-type head
and a text-output head.
