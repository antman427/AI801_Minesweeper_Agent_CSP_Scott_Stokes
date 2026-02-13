from minesweeper_env import MinesweeperEnv
import numpy as np

def print_board(obs):
    # Helper to make the numpy array look like a real board
    symbols = {
        -3: "F", # Flagged
        -2: " ", # Covered (Easier to see if blank)
        -1: "*", # Exploded Mine
        0: "."   # Empty Cell
    }

    # Column numbers
    print("   " + " ".join([str(i) for i in range(obs.shape[1])]))
    
    for r in range(obs.shape[0]):
        row_str = f"{r}  "  # Row Number
        for val in obs[r]:
            # If val is 1-8 print the number, Otherwise use the symbol.
            if val > 0:
                row_str += f"{val} "
            else:
                row_str += f"{symbols.get(val, '?')} "
        print(row_str)
    print()

# Main Game Loop
def main():
    # 1) Create the game environment
    env = MinesweeperEnv(width=9, height=9, mines=10)

    # 2) Reset to start
    obs = env.reset()

    game_over = False
    while not game_over:
        print_board(obs)

        # 3) Get User Input
        try:
            user_input = input("enter move (row col type): ").split()
            # Example input: "3 4 0" means Row 3, Col 4, Reveal
            # Type 0 = Reveal
            # Type 1 = Flag

            r = int(user_input[0])
            c = int(user_input[1])
            action_type = int(user_input[2])

            # 4) Step the environment
            obs, reward, game_over, info = env.step((r, c, action_type))

            if reward == -100:
                print("\nBOOM! You hit a mine.")
            elif reward == 100:
                print("\nYOU WIN! You cleared the board.")
            
        except (ValueError, IndexError):
            print("Invaild input. Format: row col type (eg: '0 0 0')")

    # Show Final Board
    print_board(obs)

if __name__ == "__main__":
    main()
