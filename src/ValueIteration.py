import numpy as np
from src.Affichage import Affichage

class ValueIteration:
    def __init__(self, frozen_lake, gamma=0.9):
        """
        Initialisation de l'algorithme Value Iteration.
        
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
        
        # Initialisation des valeurs d'état
        self.V = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.V[(i, j)] = 0.0
        
        # Initialisation de la Q-table
        self.Q = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.Q[(i, j)] = {action: 0.0 for action in self.actions}
                
        # Initialisation de la politique (sera calculée après value iteration)
        self.policy = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.policy[(i, j)] = {action: 0.0 for action in self.actions}
    
    def run(self, seuil=0.001, max_iterations=1000):
        """
        Exécute l'algorithme Value Iteration.
        
        Args:
            seuil: Seuil de convergence pour arrêter l'algorithme
            max_iterations: Nombre maximum d'itérations
        
        Returns:
            V: Valeurs d'état optimales
            policy: Politique optimale
        """
        iterations = 0
        
        while True:
            iterations += 1
            delta = 0  # Pour mesurer la convergence
            
            # Pour chaque état
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    state = (i, j)
                    
                    # Ignorer les états terminaux (but ou pièges)
                    if state == self.frozen_lake.goal or state in self.frozen_lake.traps:
                        continue
                    
                    # Sauvegarder l'ancienne valeur
                    v = self.V[state]
                    
                    # Initialiser la meilleure valeur d'action
                    best_action_value = float('-inf')
                    
                    # Pour chaque action possible
                    for action in self.actions:
                        # Sauvegarder l'état actuel
                        old_state = self.frozen_lake.state
                        self.frozen_lake.state = state
                        
                        # Simuler l'action et obtenir le nouvel état et la récompense
                        next_state, reward, done = self.frozen_lake.step(action)
                        
                        # Restaurer l'état
                        self.frozen_lake.state = old_state
                        
                        # Calculer la valeur de cette action
                        action_value = reward + self.gamma * self.V[next_state]
                        self.Q[state][action] = action_value
                        
                        # Mettre à jour la meilleure valeur d'action si nécessaire
                        best_action_value = max(best_action_value, action_value)
                    
                    # Mettre à jour la valeur d'état avec la meilleure valeur d'action
                    self.V[state] = best_action_value
                    
                    # Calculer le delta (différence entre l'ancienne et la nouvelle valeur)
                    delta = max(delta, abs(v - self.V[state]))
            
            # Afficher la progression toutes les 10 itérations
            if iterations % 10 == 0:
                print(f"Itération {iterations}, Delta = {delta:.6f}")
            
            # Vérifier la convergence ou le nombre maximum d'itérations
            if delta < seuil or iterations >= max_iterations:
                print(f"Algorithme convergé après {iterations} itérations avec delta = {delta:.6f}")
                break
        
        # Calculer la politique optimale à partir des valeurs d'état finales
        self._calculate_optimal_policy()
        
        return self.V, self.policy, self.Q
    
    def _calculate_optimal_policy(self):
        """
        Calcule la politique optimale basée sur les valeurs Q finales.
        """
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                
                # Ignorer les états terminaux
                if state == self.frozen_lake.goal or state in self.frozen_lake.traps:
                    continue
                
                # Trouver l'action avec la plus grande valeur Q
                best_action = max(self.Q[state], key=self.Q[state].get)
                
                # Définir la politique (déterministe)
                for action in self.actions:
                    self.policy[state][action] = 1.0 if action == best_action else 0.0
    
    def afficher_resultats(self):
        """
        Affiche les résultats de l'algorithme Value Iteration.
        """
        # Afficher la Q-table
        self.affichage.afficher_q_table(self.Q)
        
        # Afficher la politique
        self.affichage.afficher_policy(self.policy)
        
        # Afficher les valeurs d'état
        self.affichage.afficher_valeurs_etat(self.V)