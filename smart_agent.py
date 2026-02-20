from minesweeper_env import MinesweeperEnv
from minesweeper_viz import clear_screen, print_board, print_header
import random
import time


HIDDEN = -2
FLAGGED = -3


class SmartAgent:
    def __init__(self, env):
        self.env = env
        self.rows = env.height
        self.cols = env.width

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc):
                    yield nr, nc

    # ---------- CSP BUILD ----------
    def build_csp(self, obs):
        """
        Returns:
          variables: list of (r,c) hidden tiles on the frontier
          constraints: list of (vars_in_constraint, required_mines)
            where vars_in_constraint is a list of indices into variables
        """
        var_index = {}
        variables = []
        constraints = []

        for r in range(self.rows):
            for c in range(self.cols):
                if obs[r][c] > 0:  # a clue number
                    hidden_neighbors = []
                    flagged_count = 0

                    for nr, nc in self.neighbors(r, c):
                        if obs[nr][nc] == FLAGGED:
                            flagged_count += 1
                        elif obs[nr][nc] == HIDDEN:
                            if (nr, nc) not in var_index:
                                var_index[(nr, nc)] = len(variables)
                                variables.append((nr, nc))
                            hidden_neighbors.append(var_index[(nr, nc)])

                    # constraint: sum(mines among hidden_neighbors) == obs[r][c] - flagged_count
                    if hidden_neighbors:
                        required = obs[r][c] - flagged_count
                        # clamp sanity: if required is impossible, still add (will lead to no solutions)
                        constraints.append((hidden_neighbors, required))

        return variables, constraints

    # ---------- CSP SOLVER (BACKTRACKING) ----------
    def solve_csp(self, variables, constraints, max_solutions=2000):
        """
        Backtracking search for satisfying assignments.
        assignment[i] in {0,1} means variable i is safe/mine.
        Returns a list of satisfying assignments (limited).
        """
        n = len(variables)
        assignment = [-1] * n
        solutions = []

        # Precompute for quick access
        cons_vars = [cv for (cv, _) in constraints]
        cons_req  = [cr for (_, cr) in constraints]

        def consistent_partial():
            # For each constraint, check if current partial assignment can still satisfy it
            for cv, req in zip(cons_vars, cons_req):
                mines = 0
                unknown = 0
                for idx in cv:
                    val = assignment[idx]
                    if val == -1:
                        unknown += 1
                    else:
                        mines += val

                # too many mines already
                if mines > req:
                    return False
                # even if all unknown become mines, still not enough
                if mines + unknown < req:
                    return False
            return True

        def backtrack(i=0):
            if len(solutions) >= max_solutions:
                return

            if i == n:
                # full assignment, must satisfy all constraints
                for cv, req in zip(cons_vars, cons_req):
                    mines = sum(assignment[idx] for idx in cv)
                    if mines != req:
                        return
                solutions.append(assignment.copy())
                return

            # Try safe then mine (or swap order)
            for val in (0, 1):
                assignment[i] = val
                if consistent_partial():
                    backtrack(i + 1)
                assignment[i] = -1

        backtrack()
        return solutions

    def deduce_from_solutions(self, variables, solutions):
        """
        If a variable is always 0 across all solutions -> safe
        If always 1 -> mine
        """
        if not solutions:
            return set(), set()

        n = len(variables)
        always_safe = set()
        always_mine = set()

        for i in range(n):
            col = [sol[i] for sol in solutions]
            if all(v == 0 for v in col):
                always_safe.add(variables[i])
            elif all(v == 1 for v in col):
                always_mine.add(variables[i])

        return always_safe, always_mine

    # ---------- AGENT ACTION ----------
    def get_action(self, obs):
        # 1) Build CSP from current board
        variables, constraints = self.build_csp(obs)

        # If no frontier constraints, fallback to random reveal
        if not variables:
            return self.random_reveal(obs)

        # 2) Solve CSP (limited)
        solutions = self.solve_csp(variables, constraints, max_solutions=2000)

        # 3) Deduce forced moves
        always_safe, always_mine = self.deduce_from_solutions(variables, solutions)

        # Prefer flag mines first
        for (r, c) in always_mine:
            if obs[r][c] == HIDDEN:
                return (r, c, 1)  # flag

        for (r, c) in always_safe:
            if obs[r][c] == HIDDEN:
                return (r, c, 0)  # reveal
                
        # 4) # If nothing is forced, pick the tile that looks least likely to be a mine
        if solutions:
            mine_counts = [0] * len(variables)
            for sol in solutions:
                for i, v in enumerate(sol):
                    mine_counts[i] += v

            best_i = min(range(len(variables)), key=lambda i: mine_counts[i] / len(solutions))
            r, c = variables[best_i]
            if obs[r][c] == HIDDEN:
                return (r, c, 0)


        # 4) No forced move: fallback (random for now)
        return self.random_reveal(obs)

    def random_reveal(self, obs):
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                if obs[r][c] == HIDDEN:
                    moves.append((r, c))
        if not moves:
            return None
        r, c = random.choice(moves)
        return (r, c, 0)


if __name__ == "__main__":
    DELAY = 0.2

    env = MinesweeperEnv(width=9, height=9, mines=10)
    agent = SmartAgent(env)

    obs = env.reset()
    done = False
    steps = 0

    clear_screen()
    print_header("SMART AGENT (CSP)", steps, "Starting...")
    print_board(obs)
    time.sleep(1)

    while not done:
        steps += 1
        action = agent.get_action(obs)
        if action is None:
            break

        obs, reward, done, info = env.step(action)

        clear_screen()
        status_msg = f"Action: {action}"
        if done:
            status_msg = info.get("result", "Game Over").upper()

        print_header("SMART AGENT (CSP)", steps, status_msg)
        print_board(obs)

        if done:
            if reward == 100:
                print("\n>>> WIN <<<")
            else:
                print("\n>>> BOOM <<<")
            break

        time.sleep(DELAY)

