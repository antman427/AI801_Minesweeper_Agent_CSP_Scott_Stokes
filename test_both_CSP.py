from minesweeper_env import MinesweeperEnv
from csp_agent import CSPAgent
from hybrid_agent import HybridAgent
from smart_agent import SmartAgent
import time

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
    # 16x16 with 40 mines is a little difficulty
    # Small boards they all solve 100%, but large boards are harder
    # Also more mines really make the agents break
    GAMES = 100
    WIDTH = 16
    HEIGHT = 16
    MINES = 100 
    
    print(f"=== Testing Agents: {WIDTH}x{HEIGHT} Board, {MINES} Mines ===")
    
    # 1) Test Tony's Logic Agent
    csp_win, csp_time = evaluate_agent(CSPAgent, GAMES, WIDTH, HEIGHT, MINES)
    
    # 2) Test Tony's Hybrid Agent
    hyb_win, hyb_time = evaluate_agent(HybridAgent, GAMES, WIDTH, HEIGHT, MINES)
    
    # 3) Test Storm's Smart Agent
    smt_win, smt_time = evaluate_agent(SmartAgent, GAMES, WIDTH, HEIGHT, MINES)
    
    # Print Results
    print("\n" + "=" * 42)
    print(f"{'AGENT':<20} | {'WIN RATE':<10} | {'TIME/GAME':<10}")
    print("-" * 42)
    print(f"{'Tony\'s CSP':<20} | {csp_win:>8.1f}% | {csp_time:>8.4f}s")
    print(f"{'Tony\'s Hybrid':<20} | {hyb_win:>8.1f}% | {hyb_time:>8.4f}s")
    print(f"{'Storm\'s SmartAgent':<20} | {smt_win:>8.1f}% | {smt_time:>8.4f}s")
    print("=" * 42)
    
    # Print winner
    best_win_rate = max(csp_win, hyb_win, smt_win)
    if best_win_rate == hyb_win:
        print("\nCONCLUSION: Tony\'s Hybrid Agent wins!")
    elif best_win_rate == smt_win:
        print("\nCONCLUSION: Storm's Smart Agent wins!")
    elif best_win_rate == csp_win:
        print("\nCONCLUSION: Tony\'s CSP Agent wins!")
