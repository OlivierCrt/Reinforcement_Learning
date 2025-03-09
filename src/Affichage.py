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
    
    def afficher_q_table(self, Q):
        """
        Affiche la Q-table sous forme de tableau.
        """
        fig, axes = plt.subplots(1, len(self.frozen_lake.actions), figsize=(20, 5))
        fig.suptitle('Q-Table par Action', fontsize=16)
        
        for idx, action in enumerate(self.frozen_lake.actions):
            q_values = np.zeros((self.frozen_lake.grid_size, self.frozen_lake.grid_size))
            
            for i in range(self.frozen_lake.grid_size):
                for j in range(self.frozen_lake.grid_size):
                    q_values[i, j] = Q[(i, j)][action]
            
            im = axes[idx].imshow(q_values, cmap='coolwarm')
            axes[idx].set_title(f'Action: {action}')
            
            # Ajouter des étiquettes pour les cases spéciales
            for i in range(self.frozen_lake.grid_size):
                for j in range(self.frozen_lake.grid_size):
                    if (i, j) == self.frozen_lake.start:
                        axes[idx].text(j, i, 'S', ha='center', va='center', color='green', fontsize=12)
                    elif (i, j) == self.frozen_lake.goal:
                        axes[idx].text(j, i, 'G', ha='center', va='center', color='black', fontsize=12)
                    elif (i, j) in self.frozen_lake.traps:
                        axes[idx].text(j, i, 'X', ha='center', va='center', color='black', fontsize=12)
                    
                    axes[idx].text(j, i, f'{q_values[i, j]:.2f}', ha='center', va='bottom', color='black', fontsize=8)
            
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.show()
    
    def afficher_policy(self, policy):
        """
        Affiche la politique optimale sur la grille.
        """
        policy_grid = np.zeros((self.frozen_lake.grid_size, self.frozen_lake.grid_size), dtype=object)
        arrows = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }
        
        for i in range(self.frozen_lake.grid_size):
            for j in range(self.frozen_lake.grid_size):
                state = (i, j)
                if state == self.frozen_lake.goal:
                    policy_grid[i, j] = 'G'
                elif state in self.frozen_lake.traps:
                    policy_grid[i, j] = 'X'
                else:
                    best_action = max(policy[state], key=policy[state].get)
                    policy_grid[i, j] = arrows[best_action]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Politique Optimale')
        
        # Dessiner la grille
        for i in range(self.frozen_lake.grid_size):
            for j in range(self.frozen_lake.grid_size):
                if (i, j) == self.frozen_lake.goal:
                    color = 'green'
                elif (i, j) in self.frozen_lake.traps:
                    color = 'red'
                elif (i, j) == self.frozen_lake.start:
                    color = 'blue'
                else:
                    color = 'white'
                
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.3))
                ax.text(j + 0.5, i + 0.5, policy_grid[i, j], ha='center', va='center', fontsize=20)
        
        ax.set_xlim(0, self.frozen_lake.grid_size)
        ax.set_ylim(0, self.frozen_lake.grid_size)
        ax.set_xticks(np.arange(0, self.frozen_lake.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.frozen_lake.grid_size + 1, 1))
        ax.grid(True)
        ax.set_aspect('equal')
        
        # Inverser l'axe y pour avoir (0,0) en haut à gauche
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()

    def afficher_valeurs_etat(self, V):
        """
        Affiche les valeurs d'état sur la grille.
        
        Args:
            V: Dictionnaire des valeurs d'état
        """
    
        # Créer une grille des valeurs d'état
        values = np.zeros((self.frozen_lake.grid_size, self.frozen_lake.grid_size))
        for i in range(self.frozen_lake.grid_size):
            for j in range(self.frozen_lake.grid_size):
                values[i, j] = V[(i, j)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(values, cmap='viridis')
        ax.set_title('Valeurs d\'État')
        
        # Ajouter les valeurs numériques et les indicateurs spéciaux (S, G, X)
        for i in range(self.frozen_lake.grid_size):
            for j in range(self.frozen_lake.grid_size):
                text_color = 'white' if values[i, j] < 0 else 'black'
                
                # Afficher la valeur d'état
                ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', 
                        color=text_color, fontsize=9)
                
                # Ajouter des marqueurs pour les cases spéciales
                if (i, j) == self.frozen_lake.start:
                    ax.text(j, i-0.3, 'S', ha='center', va='center', 
                            color='green', fontsize=12, weight='bold')
                elif (i, j) == self.frozen_lake.goal:
                    ax.text(j, i-0.3, 'G', ha='center', va='center', 
                            color='gold', fontsize=12, weight='bold')
                elif (i, j) in self.frozen_lake.traps:
                    ax.text(j, i-0.3, 'X', ha='center', va='center', 
                            color='red', fontsize=12, weight='bold')
        
        # Ajouter une barre de couleur
        plt.colorbar(im, ax=ax)
        
        # Configurer les axes
        ax.set_xticks(np.arange(self.frozen_lake.grid_size))
        ax.set_yticks(np.arange(self.frozen_lake.grid_size))
        ax.set_xticklabels(range(self.frozen_lake.grid_size))
        ax.set_yticklabels(range(self.frozen_lake.grid_size))
        ax.set_xlabel('Colonne')
        ax.set_ylabel('Ligne')
        ax.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)
        
        plt.tight_layout()
        plt.show()
        
    def afficher_statistiques(self, rewards, steps):
        """
        Affiche les statistiques d'apprentissage: récompenses et nombre d'étapes par épisode.
        
        Args:
            rewards: Liste des récompenses par épisode
            steps: Liste du nombre d'étapes par épisode
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistiques d\'apprentissage', fontsize=16)
        
        # Graphique des récompenses
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Récompenses par épisode')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Récompense')
        
        # Graphique des étapes
        axes[0, 1].plot(steps)
        axes[0, 1].set_title('Nombre d\'étapes par épisode')
        axes[0, 1].set_xlabel('Épisode')
        axes[0, 1].set_ylabel('Nombre d\'étapes')
        
        # Moyenne mobile des récompenses (fenêtre de 100 épisodes)
        window_size = min(100, len(rewards))
        if window_size > 0:
            moving_avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(moving_avg_rewards)
            axes[1, 0].set_title(f'Moyenne mobile des récompenses (fenêtre de {window_size} épisodes)')
            axes[1, 0].set_xlabel('Épisode')
            axes[1, 0].set_ylabel('Récompense moyenne')
        
        # Moyenne mobile des étapes (fenêtre de 100 épisodes)
        if window_size > 0:
            moving_avg_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(moving_avg_steps)
            axes[1, 1].set_title(f'Moyenne mobile des étapes (fenêtre de {window_size} épisodes)')
            axes[1, 1].set_xlabel('Épisode')
            axes[1, 1].set_ylabel('Nombre moyen d\'étapes')
        
        plt.tight_layout()
        plt.show()
        
    def afficher_trajectory(self, trajectory):
        """
        Affiche la trajectoire d'un agent dans l'environnement.
        
        Args:
            trajectory: Liste de tuples (état, action, récompense)
        """
        # Créer une grille pour la trajectoire
        grid = np.zeros((self.frozen_lake.grid_size, self.frozen_lake.grid_size))
        grid[self.frozen_lake.goal] = 2  # Objectif
        for trap in self.frozen_lake.traps:
            grid[trap] = -1  # Pièges
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid, cmap='coolwarm', origin='upper')
        
        # Dessiner la trajectoire
        prev_state = None
        for state, action, reward in trajectory:
            if prev_state is not None:
                ax.arrow(prev_state[1], prev_state[0], 
                        state[1] - prev_state[1], state[0] - prev_state[0],
                        head_width=0.2, head_length=0.2, fc='black', ec='black')
            prev_state = state
        
        # Affichage des cases spéciales
        for i in range(self.frozen_lake.grid_size):
            for j in range(self.frozen_lake.grid_size):
                if (i, j) == self.frozen_lake.start:
                    ax.text(j, i, 'S', ha='center', va='center', fontsize=12, color='green')
                elif (i, j) == self.frozen_lake.goal:
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=12, color='black')
                elif (i, j) in self.frozen_lake.traps:
                    ax.text(j, i, 'X', ha='center', va='center', fontsize=12, color='black')
        
        ax.set_title('Trajectoire de l\'agent')
        ax.set_xticks(np.arange(-0.5, self.frozen_lake.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.frozen_lake.grid_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        plt.tight_layout()
        plt.show()

