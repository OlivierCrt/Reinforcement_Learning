import numpy as np

from src.FrozenLake import FrozenLake
from src.Qlearning import QLearning
from src.Affichage import Affichage

# Paramètres
grid_size = 7
episodes = 5000
alpha = 0.2
gamma = 0.9
epsilon_values = [0.001, 0.5, 0.9]  # Différentes valeurs de epsilon à tester

# Initialisation de l'environnement
env = FrozenLake(grid_size=grid_size)
display = Affichage(env)

# Boucle sur les différentes valeurs de epsilon
for epsilon in epsilon_values:
    print(f"\n--- Entraînement avec epsilon = {epsilon} ---")

    # Initialisation de l'agent Q-learning
    agent = QLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon)

    # Entraînement de l'agent
    agent.train(episodes=episodes)

    # Affichage de la Q-table
    print(f"\nQ-table pour epsilon = {epsilon}:")
    display.afficher_q_table(agent.Q)

    # Extraction et affichage de la politique optimale
    policy = agent.get_policy()
    print(f"\nPolitique optimale pour epsilon = {epsilon}:")
    display.afficher_policy(policy)