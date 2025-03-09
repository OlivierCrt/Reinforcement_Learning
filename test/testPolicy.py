from src.FrozenLake import FrozenLake
from src.PolicyIteration import PolicyIteration
from src.Affichage import Affichage

# Créer l'environnement
env = FrozenLake()

# Afficher l'environnement initial
affichage = Affichage(env)
affichage.afficher()

# Exécuter Policy Iteration
pi = PolicyIteration(env)
policy, V, Q = pi.run()

# Afficher les résultats
pi.afficher_resultats()