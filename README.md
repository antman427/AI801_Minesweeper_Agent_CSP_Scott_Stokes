# Minesweeper Agent using Constraint Satisfaction and Probabilistic Inference

A complete suite of autonomous Artificial Intelligence agents designed to solve Minesweeper. Because the Minesweeper consistency problem is mathematically proven to be NP-complete, this project benchmarks various algorithmic approaches—ranging from deterministic Constraint Satisfaction (CSP) to Markov Chain Monte Carlo (MCMC)—to evaluate performance in highly-coupled, expert-density environments.

## Project Overview

This system evaluates decision-making under extreme uncertainty by forcing AI agents to navigate "partially observable" grids. As mine density increases, the game enters a mathematical phase transition where strict logic fails, and exact probability calculations require an exponential $O(2^n)$ search space. 

This repository enables the empirical comparison of:
1. **Strict deterministic logic** using constraint satisfaction matrix algebra.
2. **Search-tree bias analysis** using depth-first search constraints.
3. **Advanced probabilistic inference** estimating risk distributions via Simulated Annealing.

### Project Demonstration Video

[![Minesweeper AI Demo](https://img.youtube.com/vi/dQw4w9WgXcQ/0.jpg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

## Key Features

* **Custom Simulation Environment**: A fast, tensor-based Minesweeper simulation API that bypasses heavy GUI overhead for high-throughput automated testing.
* **Constraint Satisfaction (CSP)**: A deduction engine leveraging subset rules and overlapping linear equations to find guaranteed safe moves.
* **Markov Chain Monte Carlo (MCMC)**: Uses simulated annealing energy minimization to calculate risk in massive, computationally intractable boundary fringes.
* **High-Throughput Benchmarking**: An automated testing suite that rapidly executes thousands of head-to-head simulations to gather stable statistical data.

## The Agents

The project contains five distinct AI architectures, each utilizing a different mathematical engine to resolve ambiguous states:

1. **Baseline CSP (`baseline_csp.py`)**: Treats the board strictly as a system of linear equations, applying matrix algebra subset rules to deduce guaranteed safes. When logic fails, it executes a structural heuristic, guessing randomly in unrevealed corners to bypass dense mine clusters.
2. **Naive CSP (`naive_csp.py`)**: Utilizes the identical CSP logic base, but falls back on a basic mathematical average of adjacent numerical clues to calculate a crude risk score when stuck.
3. **DFS Agent (`dfs_agent.py`)**: A search-based solver utilizing Depth-First Search with backtracking to find valid board states. To prevent hanging on NP-complete calculations, the search depth is capped at 2,000 solutions.
4. **Hybrid MC (`hybrid_mc.py`)**: Combines CSP logic with a Monte Carlo random sampler. For small, isolated components on the fringe (under 18 variables), it generates exact probability matrices by validating every possible binary permutation against the CSP rules.
5. **Hybrid MCMC (`hybrid_mcmc.py`)**: An advanced solver utilizing Simulated Annealing. For massive, intractable fringes, it estimates probabilities by randomly flipping variables and accepting new states based on an exponential cooling schedule, minimizing "Energy" (constraint violations).

## Project Structure

```text
AI801_Minesweeper_Agent_CSP_Scott_Stokes/
├── agents/                  # The five distinct AI architectures
│   ├── baseline_csp.py
│   ├── dfs_agent.py
│   ├── hybrid_mc.py
│   ├── hybrid_mcmc.py
│   ├── naive_csp.py
│   └── random_agent.py
├── environment/             # Core simulation and visualization
│   ├── minesweeper_env.py
│   └── minesweeper_viz.py
├── tests/                   # Benchmarking and visual generation
│   ├── benchmark.py
│   ├── generate_plots.py
│   └── test_terminal.py
├── results/                 # Output charts and performance data
│   ├── computation_times_chart.png
│   ├── test_terminal.png
│   └── win_rates_chart.png
├── README.md
└── REPORT.md
```

## Installation

### Requirements
* Python 3.8+
* NumPy >= 1.21.0
* Matplotlib >= 3.4.0

### Setup

1. Clone this repository:
```bash
git clone https://github.com/antman427/AI801_Minesweeper_Agent_CSP_Scott_Stokes.git
cd AI801_Minesweeper_Agent_CSP_Scott_Stokes
```

2. Install the required Python dependencies:
```bash
python -m pip install numpy matplotlib
```

## Usage & Execution

### 1. Watch an Agent Play (Visualizer)
The easiest way to verify the components and watch a single agent's decision-making process in real-time. The terminal maps the internal numpy arrays directly to ASCII characters to maintain low latency.

```bash
python agents/hybrid_mcmc.py
```
*(Note: You can replace `hybrid_mcmc.py` with any other agent file to watch different behaviors).*

**Terminal Legend:**
* `F` = Flagged mine
* ` ` (Blank Space) = Covered / Unrevealed cell
* `*` = Exploded mine (Loss)
* `.` = Empty revealed cell
* `1-8` = Numerical clues indicating adjacent mines

### 2. Run the Automated Benchmark
To run the high-throughput statistical benchmark across all architectures. This simulates 1,000 independent games per agent on a 16x16 grid with 60 mines (23.4% Expert Density) and outputs a comparative console table.

```bash
python tests/benchmark.py
```

### 3. Generate Data Visualizations
To automatically run the simulations and generate the `.png` bar charts mapping win rates and computation times:

```bash
python tests/generate_plots.py
```
*(Outputs will be saved directly to the `results/` directory).*

### 4. Play Manually
To test the environment manually via standard terminal `(row, col, action)` input:

```bash
python tests/test_terminal.py
```

## Experimental Results & The "Density Paradox"

Across 5,000 simulated games at Expert density, the data revealed a **"Density Paradox."** The Baseline CSP achieved the highest win rate (16.1%), outperforming the advanced Hybrid MCMC model (14.8%). 

The advanced probabilistic models allocated all computing power to finding the safest move within the highly-coupled boundary "fringe." However, at expert densities, this fringe is inherently packed with mines. The Baseline CSP, lacking probability functions, abandoned the fringe when logic failed and guessed heuristically in the dark corners of the grid. Statistically, the unrevealed open board possessed a lower average mine density than the active fringe, allowing the simpler agent to bypass dense clusters and survive longer. 

Additionally, the DFS Agent suffered from **Search Tree Bias**, achieving a 0.0% win rate due to early cutoff limits creating false 100% safety confidence.

## Course Context

This project is the final submission for **AI 801 (Foundations of Artificial Intelligence)**, integrating:
* Decision-making under uncertainty
* Constraint Satisfaction Problems (CSP)
* Search Algorithms (DFS)
* Probabilistic Reasoning (Monte Carlo / Simulated Annealing)

## Authors

* **Anthony Scott** - Pennsylvania State University (World Campus)
* **Storm Stokes** - Pennsylvania State University (World Campus)

## License

MIT License - Academic Project

## References

* Hendrickson, D., Tockman, A., & MIT Hardness Group. (2024). *Complexity of Planar Graph Orientation Consistency, Promise-Inference, and Uniqueness, with Applications to Minesweeper Variants*. 12th International Conference on Fun with Algorithms (FUN 2024). LIPIcs, Vol. 295, Article 25.
* Kaye, R. (2000). "Minesweeper is NP-complete." *The Mathematical Intelligencer*, 22(2), 9-15.
* Mehta, A. (2021). Reinforcement Learning For Constraint Satisfaction Game Agents. *arXiv preprint arXiv:2102.06019*.
* Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach (3rd ed.)*. Pearson. (Chapter 6: Constraint Satisfaction Problems; Chapter 14: Probabilistic Reasoning).
* Studholme, C. (2000). "Minesweeper as a Constraint Satisfaction Problem." Student Paper, University of Toronto.
* Wang, W., & Lei, C. (2025). Training a Minesweeper Agent Using a Convolutional Neural Network. *Applied Sciences*, 15(5), 2490. MDPI.
