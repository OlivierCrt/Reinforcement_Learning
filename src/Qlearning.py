import numpy as np
import random

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialise l'algorithme Q-Learning.
        
        Args:
            env: L'environnement FrozenLake
            alpha: Taux d'apprentissage
            gamma: Facteur de dépréciation
            epsilon: Probabilité d'exploration
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialisation de la Q-table
        self.Q = {}
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                self.Q[(i, j)] = {action: 0.0 for action in self.env.actions}
    
    def choose_action(self, state):
        """
        Choisit une action selon la politique epsilon-greedy.
        En cas d'égalité, choisit une action au hasard parmi les meilleures.
        
        Args:
            state: L'état actuel
            
        Returns:
            L'action choisie
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: action aléatoire
            return random.choice(self.env.actions)
        else:
            # Exploitation: meilleure action connue
            max_q = max(self.Q[state].values())
            best_actions = [action for action, q_value in self.Q[state].items() if q_value == max_q]
            return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, done):
        """
        Met à jour la Q-table selon l'équation de Bellman.
        
        Args:
            state: L'état actuel
            action: L'action effectuée
            reward: La récompense reçue
            next_state: L'état suivant
            done: Indique si l'épisode est terminé
        """
        # Calcul de la valeur cible pour la mise à jour
        if done:
            # Pas d'état suivant si l'épisode est terminé
            max_next_q = 0
        else:
            # Prendre la meilleure action possible pour l'état suivant
            max_next_q = max(self.Q[next_state].values())
        
        # Calcul de l'erreur de prédiction (delta)
        delta = reward + self.gamma * max_next_q - self.Q[state][action]
        
        # Mise à jour de la Q-value
        self.Q[state][action] += self.alpha * delta
    
    def train(self, episodes=1000):
        """
        Entraîne l'agent sur un nombre donné d'épisodes.
        
        Args:
            episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            rewards: Liste des récompenses par épisode
            steps: Liste du nombre d'étapes par épisode
        """
        rewards = []
        steps = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                # Choisir une action
                action = self.choose_action(state)
                
                # Effectuer l'action
                next_state, reward, done = self.env.step(action)
                
                # Mettre à jour la Q-table
                self.update(state, action, reward, next_state, done)
                
                # Mise à jour de l'état
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            
            # Affichage de la progression
            if (episode + 1) % 100 == 0:
                print(f"Épisode {episode + 1}/{episodes}, Récompense Moyenne: {np.mean(rewards[-100:]):.2f}")
        
        return rewards, steps
    
    def get_policy(self):
        """
        Extrait la politique optimale à partir de la Q-table.
        En cas d'égalité, choisit une action au hasard parmi les meilleures.
        
        Returns:
            policy: Un dictionnaire contenant la politique optimale
        """
        policy = {}
        for state in self.Q:
            # Trouver la valeur Q maximale pour cet état
            max_q = max(self.Q[state].values())
            
            # Trouver toutes les actions qui ont cette valeur Q maximale
            best_actions = [action for action, q_value in self.Q[state].items() if q_value == max_q]
            
            # Choisir une action au hasard parmi les meilleures actions
            chosen_action = random.choice(best_actions)
            
            # Définir la politique (déterministe)
            policy[state] = {action: 0.0 for action in self.env.actions}
            policy[state][chosen_action] = 1.0
        return policy
    
    def get_value_function(self):
        """
        Calcule la fonction de valeur d'état à partir de la Q-table.
        
        Returns:
            V: Un dictionnaire contenant les valeurs d'état
        """
        V = {}
        for state in self.Q:
            V[state] = max(self.Q[state].values())
        return V
    
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
            action = max(self.Q[state], key=self.Q[state].get)
            
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