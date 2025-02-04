import matplotlib.pyplot as plt
import numpy as np

class Affichage:
    def __init__(self, frozen_lake):
        self.frozen_lake = frozen_lake

    def afficher(self):
        grid = np.zeros((self.frozen_lake.grid_size, self.frozen_lake.grid_size))
        grid[self.frozen_lake.goal] = 2  # Objectif
        for trap in self.frozen_lake.traps:
            grid[trap] = -1  # Pièges

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='coolwarm', origin='upper')
        
        # Affichage des cases
        for i in range(self.frozen_lake.grid_size):
            for j in range(self.frozen_lake.grid_size):
                if (i, j) == self.frozen_lake.start:
                    ax.text(j, i, 'S', ha='center', va='center', fontsize=12, color='green')
                elif (i, j) == self.frozen_lake.goal:
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=12, color='black')
                elif (i, j) in self.frozen_lake.traps:
                    ax.text(j, i, 'X', ha='center', va='center', fontsize=12, color='black')
                elif (i, j) == self.frozen_lake.state:
                    ax.text(j, i, 'P', ha='center', va='center', fontsize=12, color='blue')
        
        ax.set_xticks(np.arange(-0.5, self.frozen_lake.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.frozen_lake.grid_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.show()