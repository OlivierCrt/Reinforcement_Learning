from src.FrozenLake import FrozenLake
from src.Qlearning import QLearning

# Créer l'environnement FrozenLake
frozen_lake = FrozenLake(grid_size=7)

# Initialiser Q-learning avec epsilon = 0.1
q_learning = QLearning(frozen_lake, epsilon=0.1)

# Entraîner l'agent
q_learning.train(num_episodes=1000)

# Afficher les résultats
q_learning.afficher_resultats()