# test_q_learning.py
import numpy as np

from src.FrozenLake import FrozenLake
from src.Qlearning import QLearning
from src.Affichage import Affichage

# Paramètres
grid_size = 7
episodes = 5000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Init
env = FrozenLake(grid_size=grid_size)
agent = QLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon)
display = Affichage(env)

# environnement initial
# print("Environnement initial:")
# display.afficher()

# Entraînement
rewards, steps = agent.train(episodes=episodes)

# Réinitialisation de l'environnement
env.reset()

# Affichage de l'environnement final
# print("\nEnvironnement après entraînement:")
# display.afficher()

# Affichage de la Q-table
print("\nQ-table:")
display.afficher_q_table(agent.Q)

# Extraction et affichage de la politique optimale
policy = agent.get_policy()
print("\nPolitique optimale:")
display.afficher_policy(policy)

# Calcul et affichage de la fonction de valeur d'état
V = agent.get_value_function()
print("\nFonction de valeur d'état:")
display.afficher_valeurs_etat(V)

# Affichage des statistiques d'apprentissage
print("\nAffichage des statistiques d'apprentissage...")
display.afficher_statistiques(rewards, steps)

# Test de la politique apprise
def test_policy(env, policy, display, episodes=10):
    total_rewards = []
    total_steps = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\nÉpisode de test {episode + 1}:")
        display.afficher()
        
        while not done:
            # Choisir la meilleure action selon la politique
            action = max(policy[state], key=policy[state].get)
            
            # Effectuer l'action
            next_state, reward, done = env.step(action)
            
            print(f"État: {state}, Action: {action}, Récompense: {reward}, Nouvel état: {next_state}")
            display.afficher()
            
            # Mise à jour de l'état
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        print(f"Récompense totale: {episode_reward}, Nombre d'étapes: {episode_steps}")
    
    return total_rewards, total_steps

# Test de la politique apprise
print("\nTest de la politique apprise:")
test_rewards, test_steps = test_policy(env, policy, display, episodes=1)