from minesweeper_env import MinesweeperEnv
from minesweeper_viz import clear_screen, print_board, print_header # <--- IMPORT HERE
import random
import time

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.rows = env.height
        self.cols = env.width

    def get_action(self, obs):
        """
        Input: obs (the board state)
        Output: (r, c, type)
        """
        # 1. Get all possible moves (cells that are currently COVERED)
        possible_moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                if obs[r, c] == -2: # -2 is Covered
                    possible_moves.append((r, c))
        
        # 2. Randomly pick one
        if not possible_moves:
            return None 
        
        move = random.choice(possible_moves)
        
        # 3. Return action: (Row, Col, Reveal=0)
        return (move[0], move[1], 0)

# --- MAIN SIMULATION LOOP ---
if __name__ == "__main__":
    # Settings
    DELAY = 0.5 
    
    # Setup
    env = MinesweeperEnv(width=9, height=9, mines=10)
    agent = RandomAgent(env)
    obs = env.reset()
    
    done = False
    steps = 0
    
    # Initial View
    clear_screen()
    print_header("Minesweeper AI", steps, "Starting...")
    print_board(obs)
    time.sleep(1)

    while not done:
        steps += 1
        
        # 1. AI Thinks
        action = agent.get_action(obs)
        
        # 2. AI Acts
        obs, reward, done, info = env.step(action)
        
        # 3. Visualization (Using our new module)
        clear_screen()
        status_msg = f"Action: Reveal {action[0]}, {action[1]}"
        if done:
            status_msg = info.get('result', 'Game Over').upper()
            
        print_header("Random Agent", steps, status_msg)
        print_board(obs)
        
        if done:
            if reward == 100: print("\n>>> CONGRATULATIONS! <<<")
            else:             print("\n>>> BOOM! <<<")
        
        time.sleep(DELAY)
