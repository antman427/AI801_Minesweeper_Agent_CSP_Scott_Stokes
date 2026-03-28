import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.benchmark import evaluate_agent
from agents.dfs_agent import DFSAgent
from agents.baseline_csp import BaselineCSPAgent
from agents.naive_csp import NaiveCSPAgent
from agents.hybrid_mc import HybridMCAgent
from agents.hybrid_mcmc import HybridMCMCAgent

def generate_live_plots(games=100, width=16, height=16, mines=60):
    print(f"=== Generating Plots: Running Benchmark ({games} games) ===")
    
    # 1) Run the agents to get live data
    dfs_win, dfs_time = evaluate_agent(DFSAgent, games, width, height, mines)
    csp_win, csp_time = evaluate_agent(BaselineCSPAgent, games, width, height, mines)
    naive_win, naive_time = evaluate_agent(NaiveCSPAgent, games, width, height, mines)
    hyb_win, hyb_time = evaluate_agent(HybridMCAgent, games, width, height, mines)
    mcmc_win, mcmc_time = evaluate_agent(HybridMCMCAgent, games, width, height, mines)

    agents = ['DFS (Limit)', 'Baseline CSP', 'Naive CSP', 'Hybrid MC', 'Hybrid MCMC']
    win_rates = [dfs_win, csp_win, naive_win, hyb_win, mcmc_win]
    times = [dfs_time, csp_time, naive_time, hyb_time, mcmc_time]

    colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#457b9d']
    
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # --- PLOT 1: Win Rates ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agents, win_rates, color=colors)
    plt.title(f'Agent Win Rates on Expert Density ({width}x{height}, {mines} Mines)', fontsize=14, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12)
    
    # Dynamically scale Y-axis based on the highest win rate
    max_win = max(win_rates)
    plt.ylim(0, max_win + 5 if max_win > 0 else 25)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_win * 0.02 + 0.2), f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/win_rates_chart.png', dpi=300)
    print("Saved results/win_rates_chart.png")

    # --- PLOT 2: Computation Time ---
    plt.figure(figsize=(10, 6))
    bars2 = plt.bar(agents, times, color=colors)
    plt.title('Average Computation Time per Game', fontsize=14, fontweight='bold')
    plt.ylabel('Time (Seconds)', fontsize=12)
    
    # Dynamically scale Y-axis based on the longest time
    max_time = max(times)
    plt.ylim(0, max_time + (max_time * 0.15) if max_time > 0 else 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_time * 0.02), f'{yval:.4f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/computation_times_chart.png', dpi=300)
    print("Saved results/computation_times_chart.png")

if __name__ == "__main__":
    generate_live_plots(games=1000, width=16, height=16, mines=60)
