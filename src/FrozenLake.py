import numpy as np


class FrozenLake:
    def __init__(self, grid_size=7):

        self.grid_size = grid_size
        self.goal = (0, 0)
        self.traps = [(2, 2), (0, 6),(4,0),(4,2),(6,3),(5,5),(4,5)]  

        self.start = (6, 6)#etat init
        self.state = self.start#etat courant
        self.actions = ['up', 'down', 'left', 'right']
        self.rewards = {
            'frozen': 0,
            'hole': -10,
            'goal': 10
        }
        self.grid = self._create_grid()
    



    def _create_grid(self):
        """On crée la grille
        """
        grid = np.zeros((self.grid_size, self.grid_size))

        grid[self.goal] = self.rewards['goal']

        for trap in self.traps:
            grid[trap] = self.rewards['hole']
        return grid
    
    def reset(self):
        """On retourne a l'etat initial"""
        self.state = self.start
        return self.state
    
    def step(self, action):
        """Effectue le mouvement et gère les bords."""
        x, y = self.state

        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.grid_size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.grid_size - 1:
            y += 1

        self.state = (x, y)

        reward = self.grid[x, y]
        finish = (self.state == self.goal) or (self.state in self.traps)  # Vérifie si on gagne ou perd

        return self.state, reward, finish