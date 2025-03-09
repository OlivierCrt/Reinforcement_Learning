import numpy as np

from src.FrozenLake import FrozenLake
from src.Qlearning import QLearning
from src.Affichage import Affichage

# Paramètres
grid_size = 7
episodes = 5000
epsilon = 0.01  # Fixé pour se concentrer sur alpha et gamma

# Combinaisons de alpha et gamma à tester
alpha_gamma_combinations = [
    (0.1, 0.99),  # Valeurs par défaut
    (0.2, 0.95),  # Apprentissage plus rapide
    (0.05, 0.99), # Apprentissage plus lent
    (0.1, 0.8),   # Vision à court terme
    (0.1, 0.95),  # Vision à long terme
]

# Initialisation de l'environnement
env = FrozenLake(grid_size=grid_size)
display = Affichage(env)

# Boucle sur les différentes combinaisons de alpha et gamma
for alpha, gamma in alpha_gamma_combinations:
    print(f"\n--- Entraînement avec alpha = {alpha}, gamma = {gamma} ---")

    # Initialisation de l'agent Q-learning
    agent = QLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon)

    # Entraînement de l'agent
    rewards, steps = agent.train(episodes=episodes)

    # Affichage de la Q-table
    print(f"\nQ-table pour alpha = {alpha}, gamma = {gamma}:")
    display.afficher_q_table(agent.Q)

    # Affichage des statistiques d'apprentissage
    print(f"\nStatistiques pour alpha = {alpha}, gamma = {gamma}:")
    print(f"Récompense moyenne sur les 100 derniers épisodes : {np.mean(rewards[-100:]):.2f}")
    print(f"Nombre moyen d'étapes sur les 100 derniers épisodes : {np.mean(steps[-100:]):.2f}")

    # Test de la politique apprise
    def test_policy(env, policy, episodes=10):
        total_rewards = []
        total_steps = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                # Choisir la meilleure action selon la politique
                action = max(policy[state], key=policy[state].get)
                
                # Effectuer l'action
                next_state, reward, done = env.step(action)
                
                # Mise à jour de l'état
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
        
        return total_rewards, total_steps

    # Test de la politique apprise
    print("\nTest de la politique apprise :")
    policy = agent.get_policy()
    test_rewards, test_steps = test_policy(env, policy, episodes=10)
    print(f"Récompenses : {test_rewards}")
    print(f"Nombre d'étapes : {test_steps}")
    print(f"Taux de réussite : {sum(1 for r in test_rewards if r > 0) / len(test_rewards) * 100:.2f}%")