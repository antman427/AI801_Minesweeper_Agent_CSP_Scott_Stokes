import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.minesweeper_env import MinesweeperEnv
from agents.baseline_csp import BaselineCSPAgent
from agents.hybrid_mc import HybridMCAgent
from agents.dfs_agent import DFSAgent
from agents.hybrid_mcmc import HybridMCMCAgent
from agents.naive_csp import NaiveCSPAgent

def evaluate_agent(agent_class, num_games=100, width=16, height=16, mines=40):
    wins = 0
    start_time = time.time()
    
    print(f"   Running {num_games} games for {agent_class.__name__}...")
    for i in range(num_games):
        env = MinesweeperEnv(width=width, height=height, mines=mines)
        agent = agent_class(env)
        obs = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(obs)
            if action is None:
                break
            obs, reward, done, info = env.step(action)
        
        if info.get('result') == 'win':
            wins += 1
            
        # Print progress every 10%
        if (i + 1) % (num_games // 10) == 0:
            print(f"      {(i+1)}/{num_games} completed...")
            
    total_time = time.time() - start_time
    win_rate = (wins / num_games) * 100
    avg_time = total_time / num_games
    
    return win_rate, avg_time

if __name__ == "__main__":
    # Test Settings
    # 16x16 with 60 mines represents "Expert" density (~23.4%)
    GAMES = 1000
    WIDTH = 16
    HEIGHT = 16
    MINES = 60
    
    print(f"=== Testing Agents: {WIDTH}x{HEIGHT} Board, {MINES} Mines ===")
    
    # 1) Test DFS Agent (Backtracking)
    dfs_win, dfs_time = evaluate_agent(DFSAgent, GAMES, WIDTH, HEIGHT, MINES)
    
    # 2) Test Baseline CSP Agent
    csp_win, csp_time = evaluate_agent(BaselineCSPAgent, GAMES, WIDTH, HEIGHT, MINES)

    # 3) Test Naive CSP Agent
    naive_win, naive_time = evaluate_agent(NaiveCSPAgent, GAMES, WIDTH, HEIGHT, MINES)
    
    # 4) Test Hybrid Monte Carlo Agent
    hyb_win, hyb_time = evaluate_agent(HybridMCAgent, GAMES, WIDTH, HEIGHT, MINES)
    
    # 5) Test Hybrid MCMC Agent (Simulated Annealing)
    mcmc_win, mcmc_time = evaluate_agent(HybridMCMCAgent, GAMES, WIDTH, HEIGHT, MINES)


    # Print Results
    print("\n" + "=" * 55)
    print(f"{'AGENT':<22} | {'WIN RATE':<10} | {'TIME/GAME':<10}")
    print("-" * 55)
    print(f"{'DFS Agent (Limit)':<22} | {dfs_win:>8.1f}% | {dfs_time:>8.4f}s")
    print(f"{'Baseline CSP':<22} | {csp_win:>8.1f}% | {csp_time:>8.4f}s")
    print(f"{'Naive CSP':<22} | {naive_win:>8.1f}% | {naive_time:>8.4f}s")
    print(f"{'Hybrid MC':<22} | {hyb_win:>8.1f}% | {hyb_time:>8.4f}s")
    print(f"{'Hybrid MCMC':<22} | {mcmc_win:>8.1f}% | {mcmc_time:>8.4f}s")
    print("=" * 55)
    
    # Print winner
    best_win_rate = max(csp_win, hyb_win, dfs_win, mcmc_win, naive_win)
    if best_win_rate == dfs_win:
        print("\nCONCLUSION: DFS Agent wins!")
    elif best_win_rate == csp_win:
        print("\nCONCLUSION: Baseline CSP Agent wins!")
    elif best_win_rate == naive_win:
        print("\nCONCLUSION: Naive CSP Agent wins!")
    elif best_win_rate == mcmc_win:
        print("\nCONCLUSION: Hybrid MCMC Agent wins!")
    elif best_win_rate == hyb_win:
        print("\nCONCLUSION: Hybrid MC Agent wins!")
