from src.FrozenLake import FrozenLake
from src.PolicyIteration import PolicyIteration
from src.Affichage import Affichage

env = FrozenLake()
affichage = Affichage(env)
affichage.afficher()
pi = PolicyIteration(env)
policy, V, Q = pi.run()
pi.afficher_resultats()