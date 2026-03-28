import random
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.minesweeper_env import MinesweeperEnv
from environment.minesweeper_viz import clear_screen, print_board, print_header


class Sentence:
    """
    A logical statement about a set of cells.
    EX: {cell_1, cell_2} = 1 means exactly one of them is a mine.
    """
    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def known_mines(self):
        """Returns the set of all cells known to be mines."""
        if len(self.cells) == self.count and self.count > 0:
            return self.cells
        return set()

    def known_safes(self):
        """Returns the set of all cells known to be safe."""
        if self.count == 0:
            return self.cells
        return set()

    def mark_mine(self, cell):
        """Updates the sentence given that 'cell' is a known mine."""
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1 

    def mark_safe(self, cell):
        """Updates the sentence given that 'cell' is a known safe."""
        if cell in self.cells:
            self.cells.remove(cell)
            # Count stays same because the mine must be in the remaining cells

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __hash__(self):
        return hash((tuple(sorted(list(self.cells))), self.count))

    def __repr__(self):
        return f"{list(self.cells)} = {self.count}"


class NaiveCSPAgent:
    def __init__(self, env):
        self.env = env
        self.rows = env.height
        self.cols = env.width

        # Knowledge Base
        self.moves_made = set()
        self.mines = set()       # Known mines
        self.safes = set()       # Known safe cells
        self.knowledge = []      # List of 'Sentence' objects
        self.knowledge_processed = set()

    def mark_mine(self, cell):
        """Updates all knowledge with a new known mine."""
        if cell in self.mines: return
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """Updates all knowledge with a new known safe cell."""
        if cell in self.safes: return
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when we reveal 'cell' and see 'count'.
        """
        # 1) Mark self as safe/move made
        self.mark_safe(cell)
        self.moves_made.add(cell)

        # 2) Find hidden neighbors
        cells = []
        mines_found = 0
        for r in range(cell[0] - 1, cell[0] + 2):
            for c in range(cell[1] - 1, cell[1] + 2):
                if (r, c) == cell: continue
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if (r, c) in self.mines:
                        mines_found += 1
                    elif (r, c) not in self.safes:
                        cells.append((r, c))

        # 3) Create Sentence
        if cells: # Only add if there are hidden neighbors
            new_sentence = Sentence(cells, count - mines_found)
            self.knowledge.append(new_sentence)

        # 4) Logic Inference Loop
        self.infer_knowledge()

    def infer_knowledge(self):
        """Iteratively clean up knowledge and deduce new facts."""
        changes = True
        while changes:
            changes = False

            # A) Check for known mines/safes in existing sentences
            safes_to_mark = set()
            mines_to_mark = set()

            for sentence in self.knowledge:
                safes_to_mark.update(sentence.known_safes())
                mines_to_mark.update(sentence.known_mines())

            if safes_to_mark or mines_to_mark:
                changes = True
                for safe in safes_to_mark: self.mark_safe(safe)
                for mine in mines_to_mark: self.mark_mine(mine)

            # B) Remove empty sentences
            self.knowledge = [s for s in self.knowledge if len(s.cells) > 0]

            # C) Subset Inference (The "Advanced" Logic)
            new_sentences = []
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1 == s2: continue

                    if s1.cells.issubset(s2.cells):
                        diff_cells = s2.cells - s1.cells # Use MINUS, not EQUALS
                        diff_count = s2.count - s1.count # Use MINUS, not EQUALS

                        if len(diff_cells) > 0:
                            new_s = Sentence(list(diff_cells), diff_count)
                            if new_s not in self.knowledge and new_s not in new_sentences:
                                new_sentences.append(new_s)
                                changes = True
            
            self.knowledge.extend(new_sentences)

    def probabilistic_guess(self, obs):
        """Choose the cell with the lowest estimated mine probability."""
    
        risk_scores = {}

        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)

                if obs[r, c] >= 0:
                    continue
                if cell in self.moves_made or cell in self.mines:
                    continue

            total_risk = 0
            contributing_clues = 0

            for sentence in self.knowledge:
                if cell in sentence.cells and len(sentence.cells) > 0:
                    risk = sentence.count / len(sentence.cells)
                    total_risk += risk
                    contributing_clues += 1

            if contributing_clues > 0:
                risk_scores[cell] = total_risk / contributing_clues
            else:
                risk_scores[cell] = 1.0  # no info → treat as risky

        if not risk_scores:
            return None

        print("[Prob Guess] Risk scores:", risk_scores)
        safest_cell = min(risk_scores, key=risk_scores.get)
        print(f"[Prob Guess] Choosing {safest_cell} with risk {risk_scores[safest_cell]:.3f}")

        # time.sleep(2)
        
        return safest_cell
        
    def get_action(self, obs):
        """Returns the best move."""
        # Sync observation with knowledge base
        for r in range(self.rows):
            for c in range(self.cols):
                if obs[r, c] >= 0:
                    if (r, c) not in self.knowledge_processed:
                        self.add_knowledge((r, c), obs[r, c])
                        self.knowledge_processed.add((r, c))
        # 1) Safe Moves
        for safe in self.safes:
            if safe not in self.moves_made:
                self.moves_made.add(safe)
                return (safe[0], safe[1], 0)
            
        # 2) Guess
        print("[CSP] No guaranteed safe move found. Using probability fallback...")
        guess = self.probabilistic_guess(obs)

        if guess and obs[guess[0], guess[1]] >= 0:
            print(f"[BUG] Tried to guess already revealed cell: {guess}")
            return None

        if guess:
            self.moves_made.add(guess)
            return (guess[0], guess[1], 0)

# --- Main Driver ---
if __name__ == "__main__":
    show_board = True
    pause_between_moves = 0.5 if show_board else 0

    wins = 0
    losses = 0
    game_num = 0

    while True:
        game_num += 1

        env = MinesweeperEnv(9, 9, 10)
        agent = NaiveCSPAgent(env)
        obs = env.reset()
        done = False
        info = {}

        if show_board:
            clear_screen()
            print_header("CSP Logic Agent", game_num, "Start")
            print_board(obs)
            # time.sleep(1)

        while not done:
            action = agent.get_action(obs)

            if action is None:
                print("[INFO] No action returned. Ending game.")
                break

            obs, reward, done, info = env.step(action)

            if show_board:
                clear_screen()
                print_header("CSP Logic Agent", game_num, f"Action: {action}")
                print_board(obs)
                # time.sleep(pause_between_moves)

        if info.get("result") == "win":
            wins += 1
            print(">>> LOGIC WINS <<<")
            print(f"[RESULT] Game {game_num}: WIN")
        else:
            losses += 1
            print(">>> LOGIC FAILED (Bad Guess) <<<")
            print(f"[RESULT] Game {game_num}: LOSS")

        total_played = wins + losses
        win_rate = (wins / total_played) * 100 if total_played > 0 else 0

        print(f"[TRACKER] Wins: {wins}")
        print(f"[TRACKER] Losses: {losses}")
        print(f"[TRACKER] Total Games: {total_played}")
        print(f"[TRACKER] Win Rate: {win_rate:.2f}%")
        print("-" * 50)

        choice = input("Press Enter to run another game, or type 'q' to quit: ").strip().lower()
        if choice == "q":
            break

    print("\n" + "=" * 50)
    print("FINAL TRACKER RESULTS")
    print("=" * 50)
    print(f"Total Games: {wins + losses}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Final Win Rate: {(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0:.2f}%")
