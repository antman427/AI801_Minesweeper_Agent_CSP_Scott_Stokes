# Algorithmic Decision-Making Under Uncertainty in Minesweeper

## Overview
This repository contains a suite of autonomous Artificial Intelligence agents designed to solve Minesweeper. Because the Minesweeper consistency problem is mathematically proven to be NP-complete, this project benchmarks various algorithmic approaches—ranging from deterministic Constraint Satisfaction (CSP) to Markov Chain Monte Carlo (MCMC)—to evaluate their performance in highly-coupled, expert-density environments.

## Repository Structure
* `/environment`: Contains the core `MinesweeperEnv` simulation API and terminal visualization tools.
* `/agents`: Contains the five distinct AI architectures evaluated in this project.
* `/tests`: Contains the automated benchmarking suite, human-playable terminal test, and plot generation scripts.
* `/results`: Contains the results from the automated benchmarking suite.

## The Agents
1. **Baseline CSP (`baseline_csp.py`)**: A deterministic logic solver utilizing algebraic subset inference.
2. **Naive CSP (`naive_csp.py`)**: A logic solver that falls back on a basic mathematical average of adjacent clues.
3. **DFS Agent (`dfs_agent.py`)**: A search-based solver utilizing Depth-First Search with backtracking to find valid board states.
4. **Hybrid MC (`hybrid_mc.py`)**: A hybrid solver utilizing Naive Monte Carlo sampling for probabilistic fallbacks on the unknown fringe.
5. **Hybrid MCMC (`hybrid_mcmc.py`)**: An advanced solver utilizing Simulated Annealing to estimate probabilities in highly-coupled, computationally intractable regions.

## Installation & Setup
**Prerequisites:** * Python 3.8+
* `numpy` (for grid tensor management)
* `matplotlib` (for generating report plots)

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/antman427/AI801_Minesweeper_Agent_CSP_Scott_Stokes.git
cd AI801_Minesweeper_Agent_CSP_Scott_Stokes
pip install numpy matplotlib
