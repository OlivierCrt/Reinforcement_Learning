from src.FrozenLake import FrozenLake
from src.Affichage import Affichage

lake = FrozenLake(grid_size=7)

affichage = Affichage(lake)

print("Grille initiale :")
affichage.afficher()
#liste d action predefinie pour le test
actions = ['up', 'right', 'right', 'down', 'right', 'up']
for action in actions:
    state, reward, done = lake.step(action)
    print(f"Action: {action}, État: {state}, Récompense: {reward}, Terminé: {done}")
    
    affichage.afficher()  

    if done:
        print("Fin du jeu.")
        break
