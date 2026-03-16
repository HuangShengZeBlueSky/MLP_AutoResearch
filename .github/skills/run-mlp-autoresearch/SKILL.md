---
name: run-mlp-autoresearch
description: 'Autonomously iterate and improve the MLP model in MLP-AutoResearch.'
argument-hint: 'Run a series of ML experiments'
user-invocable: true
---

# MLP-AutoResearch Automation

## When to Use
- When asked to autonomously run experiments and improve the MLP model in `MLP-AutoResearch`.

## Context
This is an AutoResearch framework scaled down to an MLP on MNIST. The goal is to maximize `test_accuracy` while staying within the constraints (fixed epochs, dataset).

## Procedure

For each iteration (repeat as many times as requested by the user):
1. **Analyze Data:** Read `program.md`, `runner.py`, and `train.py` to understand the current architecture and allowed changes. View `results.tsv` if it exists.
2. **Propose Modification:** Think of an improvement (e.g. optimizer, activation function, dropout, hidden size, learning rate schedule, layer normalization).
3. **Apply Code Change:** Modify `train.py` using file editing tools to implement the proposed idea. Ensure no infrastructure (like `prepare.py`) is altered.
4. **Run Experiment:** Execute `python train.py` in the terminal and wait for it to complete.
5. **Evaluate Results:** Read the output. Look for `best_test_accuracy` and compare it to previous baselines.
6. **Log Result:** Write a summary of the change, test accuracy, parameters, and whether the change was kept or reverted to `results.tsv`. If the result is worse, revert the change in `train.py`.
