import numpy as np

class FrozenLake:
    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        self.goal = (0, 0)
        self.traps = [(2, 2), (0, 6), (4, 0), (4, 2), (6, 3), (5, 5), (4, 5)]  
        self.start = (6, 6)  # état initial
        self.state = self.start  # état courant
        self.actions = ['up', 'down', 'left', 'right']
        self.rewards = {
            'frozen': 0,
            'hole': -10,
            'goal': 10
        }
        self.grid = self._create_grid()
   
    def _create_grid(self):
        """
        Crée la grille de l'environnement.
        """
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.goal] = self.rewards['goal']
        for trap in self.traps:
            grid[trap] = self.rewards['hole']
        return grid
   
    def reset(self):
        """
        Réinitialise l'environnement à son état initial.
        
        Returns:
            state: L'état initial
        """
        self.state = self.start
        return self.state
   
    def step(self, action):
        x, y = self.state
        
        # Calcul du nouvel état en fonction de l'action
        new_x, new_y = x, y
        if action == 'up':
            new_x = max(x - 1, 0)
        elif action == 'down':
            new_x = min(x + 1, self.grid_size - 1)
        elif action == 'left':
            new_y = max(y - 1, 0)
        elif action == 'right':
            new_y = min(y + 1, self.grid_size - 1)
        
        # Vérifier si l'agent a atteint un bord
        if (new_x == x and new_y == y):
            # L'agent a atteint un bord, pénalité modérée
            reward = -10  # Pénalité pour avoir atteint un bord
            done = False
        else:
            # Mise à jour de l'état
            self.state = (new_x, new_y)
            
            # Calcul de la récompense
            reward = self.grid[new_x, new_y]
            
            # Vérification si l'épisode est terminé
            done = (self.state == self.goal) or (self.state in self.traps)
        
        return self.state, reward, done