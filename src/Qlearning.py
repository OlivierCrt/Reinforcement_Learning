import numpy as np
from src.Affichage import Affichage

class QLearning:
    def __init__(self, frozen_lake, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialisation de l'algorithme Q-learning.
        
        Args:
            frozen_lake: L'environnement FrozenLake
            alpha: Taux d'apprentissage (learning rate)
            gamma: Facteur de réduction pour les récompenses futures
            epsilon: Probabilité d'exploration (epsilon-greedy)
        """
        self.frozen_lake = frozen_lake
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = frozen_lake.actions
        self.grid_size = frozen_lake.grid_size
        self.affichage = Affichage(frozen_lake)
        
        # Initialisation de la Q-table
        self.Q = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.Q[(i, j)] = {action: 0.0 for action in self.actions}
    
    def choose_action(self, state):
        """
        Choisir une action selon la stratégie epsilon-greedy.
        
        Args:
            state: État courant (tuple (i, j))
        
        Returns:
            str: Action choisie
        """
        if np.random.rand() < self.epsilon:
            # Exploration : choisir une action aléatoire
            return np.random.choice(self.actions)
        else:
            # Exploitation : choisir l'action avec la plus grande valeur Q
            return max(self.Q[state], key=self.Q[state].get)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Mettre à jour la Q-value selon la formule de Q-learning.
        
        Args:
            state: État courant (tuple (i, j))
            action: Action choisie (str)
            reward: Récompense obtenue (float)
            next_state: Prochain état (tuple (i, j))
        """
        # Valeur Q actuelle
        current_q = self.Q[state][action]
        
        # Meilleure Q-value pour le prochain état
        max_next_q = max(self.Q[next_state].values())
        
        # Mise à jour de la Q-value
        self.Q[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def train(self, num_episodes=1000):
        """
        Entraîner l'agent avec Q-learning.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
        """
        for episode in range(num_episodes):
            # Réinitialiser l'environnement
            state = self.frozen_lake.reset()
            done = False
            
            while not done:
                # Choisir une action
                action = self.choose_action(state)
                
                # Exécuter l'action et obtenir le prochain état et la récompense
                next_state, reward, done = self.frozen_lake.step(action)
                
                # Mettre à jour la Q-value
                self.update_q_value(state, action, reward, next_state)
                
                # Passer au prochain état
                state = next_state
            
            # Afficher la Q-table tous les 100 épisodes
            if (episode + 1) % 100 == 0:
                print(f"Épisode {episode + 1}")
                self.affichage.afficher_q_table(self.Q)
        
        # Afficher la Q-table finale
        print("Q-table finale :")
        self.affichage.afficher_q_table(self.Q)
    
    def get_optimal_policy(self):
        """
        Obtenir la politique optimale à partir de la Q-table.
        
        Returns:
            dict: Politique optimale (état -> action)
        """
        policy = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                policy[state] = max(self.Q[state], key=self.Q[state].get)
        return policy
    
    def afficher_resultats(self):
        """
        Afficher la Q-table et la politique optimale.
        """
        # Afficher la Q-table
        self.affichage.afficher_q_table(self.Q)
        
        # Afficher la politique optimale
        policy = self.get_optimal_policy()
        self.affichage.afficher_policy(policy)