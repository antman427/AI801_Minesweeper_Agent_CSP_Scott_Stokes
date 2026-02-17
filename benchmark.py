from minesweeper_env import MinesweeperEnv
from random_agent import RandomAgent
from csp_agent import CSPAgent
from hybrid_agent import HybridAgent
import time
import numpy as np

def run_benchmark(agent_class, num_games=100, mines=10):
    """
    Runs 'num_games' for a specific agent class and returns stats.
    """
    wins = 0
    start_time = time.time()
    
    print(f"   Running {num_games} games for {agent_class.__name__}...")
    
    for i in range(num_games):
        # Create fresh environment for every game
        env = MinesweeperEnv(width=16, height=16, mines=mines)
        agent = agent_class(env)
        obs = env.reset()
        done = False
        
        while not done:
            # We skip the print/sleep calls here for speed
            action = agent.get_action(obs)
            if action is None: break # Agent gave up
            obs, reward, done, info = env.step(action)
        
        if info.get('result') == 'win':
            wins += 1
            
        # Optional: Progress bar every 10%
        if (i+1) % (num_games // 10) == 0:
            print(f"      {i+1}/{num_games} completed...")

    total_time = time.time() - start_time
    win_rate = (wins / num_games) * 100
    avg_time = total_time / num_games
    
    return win_rate, avg_time

if __name__ == "__main__":
    # Settings
    GAMES = 1000
    MINES = 50  # Harder than standard to separate the good from the bad
    
    print(f"=== BENCHMARK STARTING ({GAMES} games, {MINES} mines) ===")
    
    # 1) Test Random
    rand_win, rand_time = run_benchmark(RandomAgent, GAMES, MINES)
    
    # 2) Test CSP (Logic Only)
    csp_win, csp_time = run_benchmark(CSPAgent, GAMES, MINES)
    
    # 3) Test Hybrid (Logic + Prob)
    hyb_win, hyb_time = run_benchmark(HybridAgent, GAMES, MINES)
    
    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print(f"{'AGENT':<15} | {'WIN RATE':<10} | {'TIME/GAME':<10}")
    print("-" * 50)
    print(f"{'Random':<15} | {rand_win:>8.1f}% | {rand_time:>8.4f}s")
    print(f"{'CSP (Logic)':<15} | {csp_win:>8.1f}% | {csp_time:>8.4f}s")
    print(f"{'Hybrid':<15} | {hyb_win:>8.1f}% | {hyb_time:>8.4f}s")
    print("="*50)
    
    print("\nCONCLUSION:")
    diff = hyb_win - csp_win
    if diff > 0:
        print(f"The Hybrid Agent improved performance by +{diff:.1f}% over pure Logic.")
    else:
        print("No significant difference found (Try increasing mine count to make it harder!)")
