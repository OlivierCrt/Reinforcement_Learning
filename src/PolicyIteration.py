import numpy as np
from src.Affichage import Affichage

class PolicyIteration:
    def __init__(self, frozen_lake, gamma=0.9):
        """
        Initialisation de l'algorithme Policy Iteration.
        
        Args:
            frozen_lake: L'environnement FrozenLake
            gamma: Facteur de réduction pour les récompenses futures
        """
        self.frozen_lake = frozen_lake
        self.gamma = gamma
        self.actions = frozen_lake.actions
        self.n_actions = len(self.actions)
        self.grid_size = frozen_lake.grid_size
        self.n_states = frozen_lake.grid_size ** 2
        self.affichage = Affichage(frozen_lake)
        
        # Initialisation de la politique (1/4)
        self.policy = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.policy[(i, j)] = {action: 1.0 / self.n_actions for action in self.actions}
        
        # Initialisation des valeurs d'état
        self.V = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.V[(i, j)] = 0.0
        
        # Initialisation q-table
        self.Q = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.Q[(i, j)] = {action: 0.0 for action in self.actions}
    
    def policy_evaluation(self, theta=0.001):
        """
        Évaluation de la politique actuelle.
        
        Args:
            theta: Seuil de convergence
        """
        delta = float('inf')
        
        while delta > theta:
            delta = 0
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    state = (i, j)
                    if state == self.frozen_lake.goal or state in self.frozen_lake.traps:
                        continue
                    
                    v = self.V[state]
                    new_v = 0
                    
                    for action in self.actions:
                        # Sauvegarder l'état actuel
                        old_state = self.frozen_lake.state
                        self.frozen_lake.state = state
                        
                        # Prendre l'action et obtenir le nouvel état et la récompense
                        next_state, reward, done = self.frozen_lake.step(action)
                        
                        # Restaurer l'état
                        self.frozen_lake.state = old_state
                        
                        # Mettre à jour la valeur selon la politique actuelle
                        new_v += self.policy[state][action] * (reward + self.gamma * self.V[next_state])
                    
                    # Mettre à jour la valeur d'état
                    self.V[state] = new_v
                    
                    # Calculer le delta pour vérifier la convergence
                    delta = max(delta, abs(v - self.V[state]))
    
    def policy_improvement(self):
        """
        Amélioration de la politique basée sur les valeurs d'état actuelles.
        
        Returns:
            bool: True si la politique a été modifiée, False sinon
        """
        policy_stable = True
        
        # Mise à jour de la Q-table
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                if state == self.frozen_lake.goal or state in self.frozen_lake.traps:
                    continue
                
                # Ancienne action avec la plus haute probabilité
                old_action = max(self.policy[state], key=self.policy[state].get)
                
                # Calculer les valeurs Q pour chaque action
                for action in self.actions:
                    # Sauvegarder l'état actuel
                    old_state = self.frozen_lake.state
                    self.frozen_lake.state = state
                    
                    # Prendre l'action et obtenir le nouvel état et la récompense
                    next_state, reward, done = self.frozen_lake.step(action)
                    
                    # Restaurer l'état
                    self.frozen_lake.state = old_state
                    
                    # Mettre à jour la Q-table
                    self.Q[state][action] = reward + self.gamma * self.V[next_state]
                
                # Trouver l'action avec la plus grande valeur Q
                best_action = max(self.Q[state], key=self.Q[state].get)
                
                # Mettre à jour la politique (déterministe)
                for action in self.actions:
                    self.policy[state][action] = 1.0 if action == best_action else 0.0
                
                # Vérifier si la politique a changé
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def run(self, max_iterations=100):
        """
        Exécuter l'algorithme Policy Iteration.
        
        Args:
            max_iterations: Nombre maximum d'itérations
        """
        for i in range(max_iterations):
            # Évaluation de la politique
            self.policy_evaluation()
            
            # Amélioration de la politique
            policy_stable = self.policy_improvement()
            
            print(f"Itération {i+1}")
            
            # Si la politique est stable, on arrête
            if policy_stable:
                print(f"Politique stable atteinte après {i+1} itérations")
                break
        
        return self.policy, self.V, self.Q
    
    def afficher_resultats(self):
        """
        Affiche la Q-table et la politique optimale.
        """
        self.affichage.afficher_q_table(self.Q)
        self.affichage.afficher_policy(self.policy)