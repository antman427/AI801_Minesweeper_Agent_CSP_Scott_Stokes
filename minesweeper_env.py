import numpy as np
import random
import sys

# Increase recursion depth for large board flood fills
sys.setrecursionlimit(2000)

class MinesweeperEnv:
    def __init__(self, width=9, height=9, mines=10):
        self.width = width
        self.height = height
        self.mines = mines
        
        # Grid: -1 = Mine, 0-8 = Count
        self.grid = None 
        
        # Visible: -2 = Covered, -3 = Flagged, True = Revealed
        self.visible = None 
        self.flags = None
        
        # State tracking
        self.mines_placed = False
        self.exploded = False
        self.action_space = [0, 1] # 0 = Reveal, 1 = Flag

    def reset(self):
        """Resets the board for a new game. Does NOT place mines yet."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.visible = np.zeros((self.height, self.width), dtype=bool)
        self.flags = np.zeros((self.height, self.width), dtype=bool)
        self.exploded = False
        self.mines_placed = False
        
        # Return observation: All -2 (Covered)
        return np.full((self.height, self.width), -2)

    def _place_mines(self, safe_r, safe_c):
        """
        Places mines randomly, ensuring (safe_r, safe_c) is SAFE.
        """
        # 1) List all possible coordinates
        coords = []
        for r in range(self.height):
            for c in range(self.width):
                # We exclude the first click so it's guaranteed safe
                if (r, c) != (safe_r, safe_c):
                    coords.append((r, c))
        
        # 2) Randomly pick mine locations
        mine_locs = random.sample(coords, self.mines)
        
        # 3) Place Mines (-1)
        for r, c in mine_locs:
            self.grid[r, c] = -1

        # 4) Calculate Numbers (0-8)
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == -1: continue
                
                # Count mines in 3x3 window
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            if self.grid[nr, nc] == -1:
                                count += 1
                self.grid[r, c] = count
        
        self.mines_placed = True

    def step(self, action):
        """
        Executes an action.
        action: (row, col, type) -> type 0=Reveal, 1=Flag
        """
        r, c, type_ = action
        
        # 0) Check Bounds
        if not (0 <= r < self.height and 0 <= c < self.width):
            return self._get_obs(), -1, False, {} # Invalid move penalty

        # 1) Handle First Move (Generate Board Now)
        if not self.mines_placed and type_ == 0:
            self._place_mines(r, c)

        # 2) Handle Flagging
        if type_ == 1:
            if not self.visible[r, c]: # Can only flag covered cells
                self.flags[r, c] = not self.flags[r, c] # Toggle
            return self._get_obs(), 0, False, {}

        # 3) Handle Reveal
        if self.flags[r, c] or self.visible[r, c]:
            return self._get_obs(), -1, False, {} # Waste of time

        # 4) Check Content
        if self.grid[r, c] == -1:
            # BOOM
            self.exploded = True
            self.visible[r, c] = True
            return self._get_obs(), -100, True, {'result': 'lose'}
        
        # 5) Safe Reveal
        self._flood_fill(r, c)
        
        # 6) Check Win Condition
        # Win if all NON-MINE cells are visible
        revealed_count = np.sum(self.visible)
        total_safe_cells = (self.width * self.height) - self.mines
        
        if revealed_count == total_safe_cells:
            return self._get_obs(), 100, True, {'result': 'win'}
        
        return self._get_obs(), 1, False, {}

    def _flood_fill(self, r, c):
        # Recursively reveals empty neighbors
        stack = [(r, c)]
        visited = set()
        visited.add((r, c))

        while stack:
            curr_r, curr_c = stack.pop()
            
            # Reveal current
            self.visible[curr_r, curr_c] = True
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    
                    nr, nc = curr_r + dr, curr_c + dc
                    
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        if not self.visible[nr, nc] and not self.flags[nr, nc] and (nr, nc) not in visited:
                            
                            # Reveal it
                            self.visible[nr, nc] = True
                            visited.add((nr, nc))
                            
                            # If it is a 0, add to stack to keep searching
                            if self.grid[nr, nc] == 0:
                                stack.append((nr, nc))

    def _get_obs(self):
        """
        Returns the board as seen by the agent.
        -2: Covered
        -3: Flagged
        -1: Exploded Mine
         0-8: Number
        """
        obs = np.full((self.height, self.width), -2)
        
        for r in range(self.height):
            for c in range(self.width):
                if self.visible[r, c]:
                    obs[r, c] = self.grid[r, c]
                elif self.flags[r, c]:
                    obs[r, c] = -3
                # Else stays -2 (Covered)
        return obs
