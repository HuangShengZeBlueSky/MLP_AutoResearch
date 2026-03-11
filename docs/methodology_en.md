# Methodology Report: AutoResearch on MLP

This document outlines the core methodology behind adapting the [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) framework to a Multi-Layer Perceptron (MLP) scenario.

## 1. System Level IPO (Input-Process-Output)

At the highest conceptual level, how does the system interact with the human user and the AI Agent?

*   **Input (I):**
    *   **Data:** MNIST dataset (fixed, provided by infrastructure).
    *   **Base Code:** Initial versions of `prepare.py` and `train.py`.
    *   **Agent Directives:** Human-written `program.md`, defining goals (maximize `test_accuracy`), budget (20 epochs), and permissible actions.
*   **Process (P):**
    *   The AI Agent repeatedly reads the current codebase and logs (`results.tsv`).
    *   It generates hypotheses about how to improve the model.
    *   It modifies `train.py`, runs the training loop, and evaluates the outcome.
    *   It makes a keep/discard decision based on the `test_accuracy` metric.
*   **Output (O):**
    *   An optimized `train.py` script containing the best found architecture and hyperparameters.
    *   An experiment log (`results.tsv`) documenting the history of explored ideas.
    *   The final, highest achieved test metric.

## 2. Experiment Loop Level IPO

What happens during a single iteration of the autonomous research loop?

*   **Input (I):**
    *   Current state of `train.py` (representing the best known configuration).
    *   Historical experiment context (knowledge of what worked and what failed from past iterations).
*   **Process (P):**
    1.  **Hypothesis Generation:** The AI proposes a change (e.g., "Replacing ReLU with GELU might improve performance").
    2.  **Code Modification:** The AI edits `train.py` to implement the hypothesis.
    3.  **Execution:** The AI runs `python train.py`.
    4.  **Evaluation:** The infrastructure (`prepare.py`) computes `final_test_accuracy`.
    5.  **Decision Making:**
        *   If new `test_accuracy` > best known `test_accuracy`: **Keep** (commit changes).
        *   Otherwise: **Discard** (rollback using `git reset --hard`).
*   **Output (O):**
    *   A new entry in `results.tsv` logging the commit hash, metrics, status (keep/discard/crash), and a brief description.
    *   Potentially, an updated "best" `train.py` codebase.

## 3. Code Module Level IPO

How are the software components structured to support this methodology?

### 3.1 `prepare.py` (The Infrastructure)

**Role:** The unchanging, objective evaluator. The AI cannot modify this file.

*   **Input:** Raw MNIST data files. Model object (provided by `train.py`).
*   **Process:**
    *   Loads and preprocesses MNIST data into `DataLoader` objects.
    *   Executes a standard training loop for exactly 20 epochs.
    *   Computes loss and accuracy on the validation/test set.
*   **Output:** Standardized performance summary (printed to stdout), focusing on `test_accuracy` and `test_loss`.

### 3.2 `train.py` (The Experiment Space)

**Role:** The canvas for the AI Agent. The only file the AI is allowed to edit.

*   **Input:** Hyperparameter values (hardcoded in the file by the AI) and the model architecture definition.
*   **Process:**
    *   `build_model()`: Constructs the PyTorch model (Plain MLP or Residual MLP) based on the current configuration.
    *   `build_optimizer()`: Initializes the optimizer (Adam, SGD, etc.) with current hyperparameter settings (lr, weight decay, etc.).
*   **Output:** Returns an instantiable PyTorch `nn.Module` and corresponding `torch.optim.Optimizer` back to the infrastructure for training.

### 3.3 `program.md` (The Prompt)

**Role:** The human-to-AI interface.

*   **Input:** Human intentions regarding constraints, metrics, and desired behavior.
*   **Process:** Translated into a structured Markdown document prioritizing clarity and unambiguous instructions for an LLM.
*   **Output:** A text prompt that initializes the AI Agent's working context and dictates its operational loop.

## 4. Design Philosophy

*   **Separation of Concerns:** Strictly dividing the *evaluator* (`prepare.py`) from the *evaluated* (`train.py`) prevents cheating and ensures fair comparisons across iterations.
*   **Fixed Budget vs. Fixed Goal:** Unlike traditional development which aims for a specific performance target regardless of time, AutoResearch uses a strictly defined budget (e.g., exactly 20 epochs) to objectively compare *efficiency* and *architecture effectiveness* under constrained resources.
*   **Greedy Optimization:** The system employs a simple greedy search strategy: only keep changes that yield immediate improvements. While this might avoid local optima, it proves highly effective in complex, high-dimensional spaces when driven by a capable LLM.
*   **Simplicity over Complexity:** The `program.md` explicitly instructs the AI to favor simpler code if performance is comparable.

## 5. Comparison: AutoResearch vs. Traditional AutoML vs. Manual Tuning

| Feature | Manual Tuning | Traditional AutoML (e.g., Optuna) | AutoResearch |
| :--- | :--- | :--- | :--- |
| **Search Space** | Limited by human intuition & time | Pre-defined, rigid (e.g., Grid/Random search over specific continuous/discrete variables) | **Open-ended, code-level.** Can invent new architectures, not just tune existing floating-point values. |
| **Driver** | Human | Statistical Algorithms (Bayesian optimization, etc.) | **LLM Agent** using reasoning and semantic understanding of code. |
| **Iterative Speed** | Slow | Fast (within defined bounds) | Fast (bounded only by model training time and API latency). |
| **Explainability** | High (Human knows *why* they changed a value) | Low (Black-box mathematical optimization) | **High.** The LLM writes commit messages (`description` in `results.tsv`) explaining *why* it made a specific code change. |
