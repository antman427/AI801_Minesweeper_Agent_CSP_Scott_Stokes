from minesweeper_env import MinesweeperEnv
from minesweeper_viz import clear_screen, print_board, print_header
from csp_agent import CSPAgent
import time
import itertools
import random

class HybridAgent(CSPAgent):
    """
    Hybrid Agent with Monte Carlo Fallback.
    1. CSP Logic (Exact)
    2. Model Counting (Exact Probability for small fringes)
    3. Monte Carlo (Approximate Probability for huge fringes)
    """

    def get_action(self, obs):
        # 1. Try strict logic first
        for r in range(self.rows):
            for c in range(self.cols):
                if obs[r, c] >= 0:
                    if (r, c) not in self.moves_made:
                        self.add_knowledge((r, c), obs[r, c])
        
        for safe in self.safes:
            if safe not in self.moves_made:
                return (safe[0], safe[1], 0)

        # 2. Logic failed? Probability Engine
        fringe_cells = set()
        for s in self.knowledge:
            fringe_cells.update(s.cells)
        
        fringe_list = list(fringe_cells)

        if not fringe_list:
            return self.get_smart_guess() 

        # Component Splitting
        components = self.get_components(fringe_list)
        
        best_overall_move = None
        best_overall_prob = 1.0

        for comp in components:
            comp_list = list(comp)
            
            # STRATEGY: 
            # Small cluster? Exact Math (Model Counting)
            # Big cluster? Approx Math (Monte Carlo)
            if len(comp) <= 18:
                move, prob = self.solve_component_exact(comp_list)
            else:
                move, prob = self.solve_component_monte_carlo(comp_list)
            
            if move and prob < best_overall_prob:
                best_overall_prob = prob
                best_overall_move = move

        # AGGRESSIVE MODE: Trust the math.
        if best_overall_move:
            # print(f"   >>> Calculated safest guess: {best_overall_move} ({best_overall_prob*100:.1f}%) <<<")
            return (best_overall_move[0], best_overall_move[1], 0)

        return self.get_smart_guess()

    def get_smart_guess(self):
        # 1. Corners
        corners = [(0, 0), (0, self.cols-1), (self.rows-1, 0), (self.rows-1, self.cols-1)]
        random.shuffle(corners)
        for r, c in corners:
            if (r, c) not in self.moves_made and (r, c) not in self.mines:
                return (r, c, 0)
        
        # 2. Edges
        edges = []
        for r in range(self.rows):
            for c in range(self.cols):
                if r==0 or r==self.rows-1 or c==0 or c==self.cols-1:
                    if (r,c) not in self.moves_made and (r,c) not in self.mines:
                        edges.append((r,c))
        if edges: return (*random.choice(edges), 0)

        # 3. Random
        possible = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.moves_made and (r,c) not in self.mines:
                    possible.append((r,c))
        if possible: return (*random.choice(possible), 0)
        return None

    def get_components(self, fringe):
        neighbors = {cell: set() for cell in fringe}
        for s in self.knowledge:
            s_cells = list(s.cells)
            for i in range(len(s_cells)):
                for j in range(i+1, len(s_cells)):
                    c1, c2 = s_cells[i], s_cells[j]
                    if c1 in neighbors and c2 in neighbors:
                        neighbors[c1].add(c2)
                        neighbors[c2].add(c1)
        
        components = []
        visited = set()
        for cell in fringe:
            if cell in visited: continue
            component = set()
            stack = [cell]
            visited.add(cell)
            while stack:
                curr = stack.pop()
                component.add(curr)
                for n in neighbors[curr]:
                    if n not in visited:
                        visited.add(n)
                        stack.append(n)
            components.append(component)
        return components

    def solve_component_exact(self, comp):
        # Exact Model Counting (Old Method)
        relevant_knowledge = []
        comp_set = set(comp)
        for s in self.knowledge:
            if not s.cells.isdisjoint(comp_set):
                relevant_knowledge.append(s)

        valid_worlds = 0
        mine_counts = {cell: 0 for cell in comp}
        
        for pattern in itertools.product([0, 1], repeat=len(comp)):
            assignment = {comp[i]: pattern[i] for i in range(len(comp))}
            if self.check_consistent(assignment, relevant_knowledge):
                valid_worlds += 1
                for cell in comp:
                    if assignment[cell] == 1: mine_counts[cell] += 1

        if valid_worlds == 0: return None, 1.0
        
        best_prob = 1.0
        best_cell = None
        for cell in comp:
            prob = mine_counts[cell] / valid_worlds
            if prob < best_prob:
                best_prob = prob
                best_cell = cell
        return best_cell, best_prob

    def solve_component_monte_carlo(self, comp):
        # Approx Model Counting (New Method for Big Clusters)
        relevant_knowledge = []
        comp_set = set(comp)
        for s in self.knowledge:
            if not s.cells.isdisjoint(comp_set):
                relevant_knowledge.append(s)

        valid_worlds = 0
        mine_counts = {cell: 0 for cell in comp}
        
        # Try 5000 random samples
        samples = 5000
        for _ in range(samples):
            # Random binary assignment
            assignment = {cell: random.choice([0, 1]) for cell in comp}
            
            if self.check_consistent(assignment, relevant_knowledge):
                valid_worlds += 1
                for cell in comp:
                    if assignment[cell] == 1: mine_counts[cell] += 1
        
        if valid_worlds == 0:
            # If random sampling failed to find ANY valid world, 
            # it's too complex. Return None so we pick a random move.
            return None, 1.0

        best_prob = 1.0
        best_cell = None
        for cell in comp:
            prob = mine_counts[cell] / valid_worlds
            if prob < best_prob:
                best_prob = prob
                best_cell = cell
        return best_cell, best_prob

    def check_consistent(self, assignment, sentences):
        for s in sentences:
            mine_count = 0
            unknowns = 0
            for cell in s.cells:
                if cell in assignment:
                    mine_count += assignment[cell]
                else:
                    unknowns += 1
            
            # Strict check if fully assigned
            if unknowns == 0:
                if mine_count != s.count: return False
            
            # Partial check
            if mine_count > s.count: return False
        return True
    
# --- Main Driver ---
if __name__ == "__main__":
    # Settings: Higher mine count to force the AI to guess more often
    env = MinesweeperEnv(9, 9, 12)
    agent = HybridAgent(env)
    obs = env.reset()
    done = False

    clear_screen()
    print_header("Hybrid Probabilistic Agent", 0, "Start")
    print_board(obs)
    input("Press Enter to Start...")

    while not done:
        action = agent.get_action(obs)
        if action is None: break

        obs, reward, done, info = env.step(action)

        clear_screen()
        print_header("Hybrid Agent", 0, f"Action: {action}")
        print_board(obs)

        if info.get('result') == 'win':
            print("\n>>> HYBRID AGENT WINS! <<<")
        else:
            print("\n>>> HIT MINE (Unluckly Guess) <<<")

        # time.sleep(0.2)  # Faster speed
