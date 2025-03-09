import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.FrozenLake import FrozenLake
from src.PolicyIteration import PolicyIteration

def test_policy_iteration():
    # Créer l'environnement FrozenLake
    env = FrozenLake(grid_size=7)
    
    # Initialiser l'algorithme de Policy Iteration
    pi = PolicyIteration(env, gamma=0.9, theta=1e-6)
    
    # Exécuter l'algorithme
    print("Exécution de l'algorithme Policy Iteration...")
    optimal_policy = pi.run(max_iterations=100)
    
    # Afficher la Q-table
    pi.display_q_table()
    
    # Afficher la politique optimale
    pi.display_policy()
    
    # Visualiser la fonction valeur
    print("\nAffichage de la fonction valeur...")
    pi.plot_value_function()
    
    # Visualiser la politique optimale
    print("\nAffichage de la politique optimale...")
    pi.plot_policy()
    
    # Tester la politique optimale
    print("\nTest de la politique optimale:")
    trajectory = pi.record_trajectory(max_steps=100)
    
    # Afficher la trajectoire
    states = [t[0] for t in trajectory[:-1]]  # Sans l'état final
    actions = [t[1] for t in trajectory[:-1]]
    rewards = [t[2] for t in trajectory[:-1]]
    final_state = trajectory[-1][0]
    
    print(f"État initial: {states[0]}")
    for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
        print(f"Étape {i+1}: État {state}, Action {action}, Récompense {reward}")
    print(f"État final: {final_state}")
    
    total_reward = sum(rewards)
    print(f"Récompense totale: {total_reward}")
    print(f"Nombre d'étapes: {len(states)}")
    
    # Visualisation du chemin parcouru
    visualize_trajectory(env, states + [final_state])

def visualize_trajectory(env, path):
    """
    Visualise une trajectoire sur une grille.
    
    Args:
        env: L'environnement FrozenLake
        path: Liste des états visités
    """
    grid = np.zeros((env.grid_size, env.grid_size))
    
    # Marquer les trous
    for trap in env.traps:
        grid[trap] = -1
    
    # Marquer l'objectif
    grid[env.goal] = 2
    
    # Marquer le chemin parcouru
    for i, pos in enumerate(path):
        if grid[pos] == 0:  # Ne pas écraser les trous ou l'objectif
            grid[pos] = 1
    
    # Création d'une palette de couleurs personnalisée
    cmap = plt.cm.colors.ListedColormap(['lightblue', 'lightgreen', 'gold', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap=cmap, norm=norm)
    
    # Ajouter les annotations pour chaque cellule
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            cell_text = ""
            if (i, j) == env.start:
                cell_text = "S"
            elif (i, j) == env.goal:
                cell_text = "G"
            elif (i, j) in env.traps:
                cell_text = "H"
            
            # Ajouter un numéro pour chaque étape du chemin
            if (i, j) in path:
                step_num = path.index((i, j))
                if cell_text:
                    cell_text += f"\n{step_num}"
                else:
                    cell_text = str(step_num)
            
            plt.text(j, i, cell_text, ha="center", va="center", color="black", fontweight="bold")
    
    # Ajouter la grille
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    plt.title("Chemin parcouru avec la politique optimale")
    plt.xticks(np.arange(-.5, env.grid_size, 1), [])
    plt.yticks(np.arange(-.5, env.grid_size, 1), [])
    plt.show()

if __name__ == "__main__":
    test_policy_iteration()