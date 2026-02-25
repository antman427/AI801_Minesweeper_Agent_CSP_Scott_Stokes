import numpy as np
import random

class MinesweeperEnv:
    def __init__(self, width = 9, height = 9, mines = 10):
        self.width = width
        self.height = height
        self.mines = mines
        self.grid = np.zeros((height, width), dtype=int)      # -1 = Mine, 0-8 = Hints
        self.visible = np.zeros((height, width), dtype=bool)  # False = Covered
        self.flags = np.zeros((height, width), dtype=bool)
        self.game_over = False
        self.won = False
        self.reset()

    def reset(self):
        # Resets the board and places mines randomly
        self.grid.fill(0)
        self.visible.fill(False)
        self.flags.fill(False)
        self.game_over = False
        self.won = False

        # Place mines
        mine_indexs = random.sample(range(self.width * self.height), self.mines)
        for idx in mine_indexs:
            r, c = divmod(idx, self.width)
            self.grid[r, c] = -1

        # Calculate neighbor hints
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == -1:
                    continue
                self.grid[r, c] = self._count_mines(r, c)

        return self._get_observation()
    
    def _count_mines(self, r, c):
        # Counts mines around (r, c)
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if self.grid[nr, nc] == -1:
                        count += 1
        return count
    
    def step(self, action):
        # Action: (r, c, type) where type 0 = Reveal and 0 = Flag
        # Returns: observation, reward, done, info
        r, c, type = action

        if self.game_over:
            return self._get_observation(), 0, True, {}
        
        # Flag Action
        if type == 1:
            self.flags[r, c] = not self.flags[r, c]
            return self._get_observation(), 0, False, {}
        
        # Reveal Action
        if self.flags[r, c]:
            # Cannot reveal flagged
            return self._get_observation(), -1, False, {}
        
        # Hit Mine
        if self.grid[r, c] == -1:
            self.game_over = True
            self.visible[r, c] = True
            return self._get_observation(), -100, True, {"result": "lose"}
        
        # Safe Reveal
        self.visible[r, c] = True
        reward = 1

        if self.grid[r, c] == 0:
            # Recursive clear
            self._flood_fill(r, c)

        # Check Win Condition
        if np.sum(self.visible) == (self.width * self.height) - self.mines:
            self.game_over = True
            self.won = True
            reward = 100
            return self._get_observation(), 100, True, {"result": "win"}
        
        return self._get_observation(), reward, False, {}
        
    def _flood_fill(self, r, c):
        # Recursively reveals empty neighbors
        stack = [(r, c)]

        # Keep track of what was processed in this fill to avoid loops
        visited = set()
        visited.add((r,c))

        while stack:
            curr_r, curr_c = stack.pop()

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue  # Skip self

                    nr, nc = curr_r + dr, curr_c + dc

                    # Check again
                    if 0 <= nr < self.height and 0 <= nc < self.width:

                        # Only process if NOT visible, NOT Flagged, and NOT alreday visited
                        if not self.visible[nr, nc] and not self.flags[nr, nc] and (nr, nc) not in visited:
                            
                            # Reveal it
                            self.visible[nr, nc] = True
                            visited.add((nr, nc))  # Mark as processed

                            # If it is a 0, add to stack to keep searching
                            if self.grid[nr, nc] == 0:
                                stack.append((nr, nc))

    def _get_observation(self):
        # Returns the board from the agent's perspective (-2 for covered)
        obs = np.full((self.height, self.width), -2)  # -2 = Covered
        obs[self.visible] = self.grid[self.visible]
        obs[self.flags] = -3  # -3 = Flagged
        return obs
