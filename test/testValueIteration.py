from src.FrozenLake import FrozenLake
from src.ValueIteration import ValueIteration
from src.Affichage import Affichage

# Créer l'environnement FrozenLake
env = FrozenLake()

# Afficher l'environnement initial
affichage = Affichage(env)
affichage.afficher()

# Créer et exécuter l'algorithme Value Iteration
value_iter = ValueIteration(env)
V, policy, Q = value_iter.run(seuil=0.001, max_iterations=1000)

# Afficher les résultats
value_iter.afficher_resultats()

# Vous pouvez également tester avec différentes valeurs de seuil pour comparer
# Par exemple:
# V_loose, policy_loose, Q_loose = value_iter.run(seuil=0.01)
# V_strict, policy_strict, Q_strict = value_iter.run(seuil=0.0001)