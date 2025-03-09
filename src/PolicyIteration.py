import numpy as np
import matplotlib.pyplot as plt

class PolicyIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        """
        Initialise l'algorithme de Policy Iteration.
        
        Args:
            env: L'environnement FrozenLake
            gamma: Facteur d'actualisation pour les récompenses futures
            theta: Seuil de convergence pour l'évaluation de la politique
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.states = [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]
        self.n_states = env.grid_size * env.grid_size
        self.n_actions = len(env.actions)
        self.actions = env.actions
        
        # Initialisation de la valeur des états à zéro
        self.V = {state: 0 for state in self.states}
        
        # Initialisation de la politique à des actions aléatoires
        self.policy = {state: np.random.choice(self.actions) for state in self.states}
        
        # Initialisation de la Q-table
        self.Q = {state: {action: 0 for action in self.actions} for state in self.states}
    
    def policy_evaluation(self):
        """
        Évalue la politique actuelle en calculant la fonction valeur.
        """
        delta = float('inf')
        
        while delta > self.theta:
            delta = 0
            
            for state in self.states:
                old_value = self.V[state]
                
                # Si l'état est terminal (trou ou objectif), sa valeur reste à 0
                if state == self.env.goal or state in self.env.traps:
                    self.V[state] = self.env.grid[state]
                    continue
                
                action = self.policy[state]
                new_value = 0
                
                # Sauvegarde de l'état actuel de l'environnement
                old_state = self.env.state
                self.env.state = state
                
                next_state, reward, done = self.env.step(action)
                
                # Mise à jour de la valeur de l'état
                new_value = reward + self.gamma * self.V[next_state] * (not done)
                
                # Restauration de l'état de l'environnement
                self.env.state = old_state
                
                self.V[state] = new_value
                delta = max(delta, abs(old_value - new_value))
    
    def policy_improvement(self):
        """
        Améliore la politique en choisissant les actions qui maximisent la valeur.
        
        Returns:
            bool: True si la politique a été modifiée, False sinon
        """
        policy_stable = True
        
        for state in self.states:
            # Si l'état est terminal, pas besoin de mettre à jour la politique
            if state == self.env.goal or state in self.env.traps:
                continue
            
            old_action = self.policy[state]
            
            # Calcul des valeurs Q pour chaque action
            q_values = {}
            for action in self.actions:
                # Sauvegarde de l'état actuel de l'environnement
                old_state = self.env.state
                self.env.state = state
                
                next_state, reward, done = self.env.step(action)
                
                # Calcul de la valeur Q pour cette action
                q_values[action] = reward + self.gamma * self.V[next_state] * (not done)
                
                # Mise à jour de la Q-table
                self.Q[state][action] = q_values[action]
                
                # Restauration de l'état de l'environnement
                self.env.state = old_state
            
            # Choisir l'action qui maximise la valeur Q
            best_action = max(q_values, key=q_values.get)
            self.policy[state] = best_action
            
            # Vérifier si la politique a changé
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def run(self, max_iterations=1000):
        """
        Exécute l'algorithme de Policy Iteration.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            dict: La politique optimale
        """
        for i in range(max_iterations):
            # Évaluation de la politique
            self.policy_evaluation()
            
            # Amélioration de la politique
            policy_stable = self.policy_improvement()
            
            print(f"Itération {i+1} terminée")
            
            # Si la politique est stable, on a convergé
            if policy_stable:
                print(f"Politique stable après {i+1} itérations")
                break
        
        return self.policy
    
    def get_policy(self):
        """
        Extrait la politique optimale sous format probabiliste comme dans QLearning.
        
        Returns:
            policy: Un dictionnaire contenant la politique optimale
        """
        policy_prob = {}
        for state in self.states:
            policy_prob[state] = {action: 0.0 for action in self.actions}
            policy_prob[state][self.policy[state]] = 1.0
        return policy_prob
    
    def get_value_function(self):
        """
        Retourne la fonction de valeur d'état.
        
        Returns:
            V: Un dictionnaire contenant les valeurs d'état
        """
        return self.V
    
    def display_q_table(self):
        """
        Affiche la Q-table.
        """
        print("\nQ-table:")
        for state in self.states:
            print(f"État {state}:")
            for action in self.actions:
                print(f"  {action}: {self.Q[state][action]:.2f}")
    
    def display_policy(self):
        """
        Affiche la politique sous forme de grille.
        """
        policy_grid = np.empty((self.env.grid_size, self.env.grid_size), dtype=object)
        
        # Conversion des actions en flèches pour une meilleure visualisation
        action_arrows = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }
        
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                state = (i, j)
                if state == self.env.goal:
                    policy_grid[i, j] = 'G'  # Goal
                elif state in self.env.traps:
                    policy_grid[i, j] = 'H'  # Hole
                else:
                    policy_grid[i, j] = action_arrows[self.policy[state]]
        
        print("\nPolitique optimale:")
        for i in range(self.env.grid_size):
            row = ' '.join(policy_grid[i])
            print(row)
    
    def record_trajectory(self, max_steps=100):
        """
        Enregistre une trajectoire en suivant la politique optimale actuelle.
        
        Args:
            max_steps: Nombre maximum d'étapes
            
        Returns:
            trajectory: Liste de tuples (état, action, récompense)
        """
        state = self.env.reset()
        trajectory = []
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Choisir la meilleure action selon la politique actuelle
            action = self.policy[state]
            
            # Effectuer l'action
            next_state, reward, done = self.env.step(action)
            
            # Enregistrer l'étape
            trajectory.append((state, action, reward))
            
            # Mise à jour de l'état
            state = next_state
            steps += 1
            
        # Ajouter l'état final
        trajectory.append((state, None, None))
        
        return trajectory
    
    def plot_value_function(self):
        """
        Affiche la fonction valeur sous forme de heatmap.
        """
        value_grid = np.zeros((self.env.grid_size, self.env.grid_size))
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                value_grid[i, j] = self.V[(i, j)]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(value_grid, cmap="YlGnBu")
        plt.colorbar(label="Valeur")
        
        # Ajouter les annotations pour chaque cellule
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                plt.text(j, i, f"{value_grid[i, j]:.2f}", ha="center", va="center", color="black")
        
        plt.title("Fonction valeur")
        plt.show()
    
    def plot_policy(self):
        """
        Visualise la politique sous forme de heatmap colorée.
        """
        # Conversion des actions en flèches pour une meilleure visualisation
        action_arrows = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }
        
        policy_grid = np.empty((self.env.grid_size, self.env.grid_size), dtype=object)
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                state = (i, j)
                if state == self.env.goal:
                    policy_grid[i, j] = 'G'  # Goal
                elif state in self.env.traps:
                    policy_grid[i, j] = 'H'  # Hole
                else:
                    policy_grid[i, j] = action_arrows[self.policy[state]]
        
        # Création d'une heatmap colorée pour visualiser la politique
        # Utilisation de codes couleur pour les différentes actions
        action_codes = {
            'up': 0,
            'right': 1,
            'down': 2,
            'left': 3,
            'G': 4,  # Goal
            'H': 5   # Hole
        }
        
        policy_color_grid = np.zeros((self.env.grid_size, self.env.grid_size))
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                state = (i, j)
                if state == self.env.goal:
                    policy_color_grid[i, j] = action_codes['G']
                elif state in self.env.traps:
                    policy_color_grid[i, j] = action_codes['H']
                else:
                    policy_color_grid[i, j] = action_codes[self.policy[state]]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(policy_color_grid, cmap="coolwarm")
        
        # Ajouter les annotations pour chaque cellule
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                plt.text(j, i, policy_grid[i, j], ha="center", va="center", color="black")
        
        plt.title("Politique optimale")
        plt.show()