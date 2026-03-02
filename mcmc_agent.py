from minesweeper_env import MinesweeperEnv
from csp_agent import CSPAgent
import time
import itertools
import random
import math

class MCMCAgent(CSPAgent):
    """
    Advanced Agent utilizing Markov Chain Monte Carlo (Simulated Annealing)
    to estimate probabilities for massively constrained state spaces.
    """

    def get_action(self, obs):
        # 1) Try strict logic first
        for r in range(self.rows):
            for c in range(self.cols):
                if obs[r, c] >= 0:
                    if (r, c) not in self.moves_made:
                        self.add_knowledge((r, c), obs[r, c])
        
        for safe in self.safes:
            if safe not in self.moves_made:
                return (safe[0], safe[1], 0)

        # 2) If logic fails, use Probability Engine
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
            
            # STRATEGY: Exact Math for small clusters, MCMC for massive clusters
            if len(comp) <= 18:
                move, prob = self.solve_component_exact(comp_list)
            else:
                move, prob = self.solve_component_mcmc(comp_list)
            
            if move and prob < best_overall_prob:
                best_overall_prob = prob
                best_overall_move = move

        # AGGRESSIVE MODE: Trust the math
        if best_overall_move:
            return (best_overall_move[0], best_overall_move[1], 0)

        return self.get_smart_guess()

    def solve_component_mcmc(self, comp):
        """
        Uses Simulated Annealing to sample the state space of large CSP components
        """
        relevant_knowledge = []
        comp_set = set(comp)
        for s in self.knowledge:
            if not s.cells.isdisjoint(comp_set):
                relevant_knowledge.append(s)

        # MCMC Tuning Parameters
        MAX_STEPS = 15000
        T_START = 2.0
        T_END = 0.01

        # Initialize a random state (0 = Safe, 1 = Mine)
        state = {cell: random.choice([0, 1]) for cell in comp}

        def calc_energy(st):
            # Energy = sum of absolute errors in constraint equations
            energy = 0
            for s in relevant_knowledge:
                mines = sum(st.get(c, 0) for c in s.cells if c in st)
                energy += abs(mines - s.count)
            return energy

        current_energy = calc_energy(state)
        
        valid_samples = 0
        mine_counts = {cell: 0 for cell in comp}

        for step in range(MAX_STEPS):
            # Exponential cooling schedule
            T = T_START * ((T_END / T_START) ** (step / MAX_STEPS))
            if T == 0: T = 0.0001
            
            # Propose a state change: flip one random variable
            cell_to_flip = random.choice(comp)
            state[cell_to_flip] = 1 - state[cell_to_flip]
            
            new_energy = calc_energy(state)
            
            # Metropolis-Hastings Acceptance Criterion
            if new_energy <= current_energy:
                current_energy = new_energy
            else:
                prob_accept = math.exp(-(new_energy - current_energy) / T)
                if random.random() < prob_accept:
                    current_energy = new_energy
                else:
                    # Reject proposal, flip it back
                    state[cell_to_flip] = 1 - state[cell_to_flip]

            # If we are in a valid world (Energy = 0), collect the sample!
            if current_energy == 0:
                valid_samples += 1
                for c in comp:
                    if state[c] == 1:
                        mine_counts[c] += 1

        if valid_samples == 0:
            return None, 1.0

        best_prob = 1.0
        best_cell = None
        for cell in comp:
            prob = mine_counts[cell] / valid_samples
            if prob < best_prob:
                best_prob = prob
                best_cell = cell

        return best_cell, best_prob

    # --- Keep exact solver, components, and heuristics from HybridAgent ---
    def solve_component_exact(self, comp):
        relevant_knowledge = []
        comp_set = set(comp)
        for s in self.knowledge:
            if not s.cells.isdisjoint(comp_set):
                relevant_knowledge.append(s)

        valid_worlds = 0
        mine_counts = {cell: 0 for cell in comp}
        for pattern in itertools.product([0, 1], repeat=len(comp)):
            assignment = {comp[i]: pattern[i] for i in range(len(comp))}
            
            consistent = True
            for s in relevant_knowledge:
                mines = sum(assignment.get(c, 0) for c in s.cells if c in assignment)
                unknowns = sum(1 for c in s.cells if c not in assignment)
                if unknowns == 0 and mines != s.count: consistent = False; break
                if mines > s.count: consistent = False; break
            
            if consistent:
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

    def get_smart_guess(self):
        corners = [(0, 0), (0, self.cols-1), (self.rows-1, 0), (self.rows-1, self.cols-1)]
        random.shuffle(corners)
        for r, c in corners:
            if (r, c) not in self.moves_made and (r, c) not in self.mines:
                return (r, c, 0)
        edges = [(r, c) for r in range(self.rows) for c in range(self.cols) if r==0 or r==self.rows-1 or c==0 or c==self.cols-1]
        valid_edges = [(r, c) for r, c in edges if (r, c) not in self.moves_made and (r, c) not in self.mines]
        if valid_edges: return (*random.choice(valid_edges), 0)
        possible = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.moves_made and (r, c) not in self.mines]
        if possible: return (*random.choice(possible), 0)
        return None
