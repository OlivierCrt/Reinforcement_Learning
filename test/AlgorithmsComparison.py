import numpy as np
import matplotlib.pyplot as plt
from src.Affichage import Affichage
from src.FrozenLake import FrozenLake
from src.ValueIteration import ValueIteration
from src.PolicyIteration import PolicyIteration
from src.Qlearning import QLearning

class AlgorithmsComparison:
    def __init__(self, grid_size=7):
        """
        Initialise la comparaison des algorithmes.
        
        Args:
            grid_size: Taille de la grille
        """
        self.env = FrozenLake(grid_size=grid_size)
        self.affichage = Affichage(self.env)
        
        # Paramètres communs
        self.gamma = 0.9
        
        # Initialisation des algorithmes
        self.value_iteration = ValueIteration(self.env, gamma=self.gamma)
        self.policy_iteration = PolicyIteration(self.env, gamma=self.gamma)
        
        # Paramètres pour Q-Learning
        self.alpha = 0.1
        self.epsilon_values = [0.1, 0.2, 0.5]
        self.q_learning_agents = {}
        
        for epsilon in self.epsilon_values:
            self.q_learning_agents[epsilon] = QLearning(
                self.env, alpha=self.alpha, gamma=self.gamma, epsilon=epsilon
            )
    
    def run_all_algorithms(self, episodes=1000, vi_theta=0.001, pi_theta=0.001):
        """
        Exécute tous les algorithmes et mesure leurs performances.
        
        Args:
            episodes: Nombre d'épisodes pour Q-Learning
            vi_theta: Seuil de convergence pour Value Iteration
            pi_theta: Seuil de convergence pour Policy Iteration
            
        Returns:
            results: Dictionnaire contenant les résultats
        """
        results = {}
        
        # Exécuter Value Iteration
        print("Exécution de Value Iteration...")
        vi_start_time = np.datetime64('now')
        v_vi, policy_vi, q_vi = self.value_iteration.run(seuil=vi_theta)
        vi_end_time = np.datetime64('now')
        vi_time = (vi_end_time - vi_start_time) / np.timedelta64(1, 's')
        results['value_iteration'] = {
            'time': vi_time,
            'values': v_vi,
            'policy': policy_vi,
            'q_values': q_vi
        }
        
        # Exécuter Policy Iteration
        print("Exécution de Policy Iteration...")
        pi_start_time = np.datetime64('now')
        policy_pi, v_pi, q_pi = self.policy_iteration.run()
        pi_end_time = np.datetime64('now')
        pi_time = (pi_end_time - pi_start_time) / np.timedelta64(1, 's')
        results['policy_iteration'] = {
            'time': pi_time,
            'values': v_pi,
            'policy': policy_pi,
            'q_values': q_pi
        }
        
        # Exécuter Q-Learning avec différentes valeurs d'epsilon
        results['q_learning'] = {}
        
        for epsilon in self.epsilon_values:
            print(f"Exécution de Q-Learning avec epsilon={epsilon}...")
            ql_start_time = np.datetime64('now')
            rewards, steps = self.q_learning_agents[epsilon].train(episodes=episodes)
            ql_end_time = np.datetime64('now')
            ql_time = (ql_end_time - ql_start_time) / np.timedelta64(1, 's')
            
            policy = self.q_learning_agents[epsilon].get_policy()
            values = self.q_learning_agents[epsilon].get_value_function()
            
            results['q_learning'][epsilon] = {
                'time': ql_time,
                'rewards': rewards,
                'steps': steps,
                'policy': policy,
                'values': values,
                'q_values': self.q_learning_agents[epsilon].Q
            }
        
        return results
    
    def calculate_policy_similarity(self, policy1, policy2):
        """
        Calcule la similarité entre deux politiques.
        
        Args:
            policy1: Première politique
            policy2: Deuxième politique
            
        Returns:
            similarity: Pourcentage de similarité
        """
        count_same = 0
        count_total = 0
        
        for state in policy1:
            if state == self.env.goal or state in self.env.traps:
                continue
                
            best_action1 = max(policy1[state], key=policy1[state].get)
            best_action2 = max(policy2[state], key=policy2[state].get)
            
            if best_action1 == best_action2:
                count_same += 1
            
            count_total += 1
        
        return (count_same / count_total) * 100 if count_total > 0 else 0
    
    def display_comparison(self, results):
        """
        Affiche une comparaison visuelle des résultats.
        
        Args:
            results: Dictionnaire contenant les résultats
        """
        # Comparaison des temps d'exécution
        algorithms = ['Value Iteration', 'Policy Iteration'] + [f'Q-Learning (ε={e})' for e in self.epsilon_values]
        times = [
            results['value_iteration']['time'],
            results['policy_iteration']['time']
        ]
        times.extend([results['q_learning'][eps]['time'] for eps in self.epsilon_values])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(algorithms, times)
        ax.set_title('Temps d\'exécution des algorithmes')
        ax.set_ylabel('Temps (secondes)')
        ax.set_xlabel('Algorithme')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        plt.show()
        
        # Comparaison de la similarité des politiques
        policy_vi = results['value_iteration']['policy']
        policy_pi = results['policy_iteration']['policy']
        
        print(f"Similarité entre Value Iteration et Policy Iteration: {self.calculate_policy_similarity(policy_vi, policy_pi):.2f}%")
        
        for epsilon in self.epsilon_values:
            policy_ql = results['q_learning'][epsilon]['policy']
            similarity_vi_ql = self.calculate_policy_similarity(policy_vi, policy_ql)
            similarity_pi_ql = self.calculate_policy_similarity(policy_pi, policy_ql)
            
            print(f"Similarité entre Value Iteration et Q-Learning (ε={epsilon}): {similarity_vi_ql:.2f}%")
            print(f"Similarité entre Policy Iteration et Q-Learning (ε={epsilon}): {similarity_pi_ql:.2f}%")
        
        # Comparaison des performances de Q-Learning avec différentes valeurs d'epsilon
        plt.figure(figsize=(15, 10))
        
        # Récompenses moyennes
        plt.subplot(2, 1, 1)
        for epsilon in self.epsilon_values:
            rewards = results['q_learning'][epsilon]['rewards']
            window_size = min(100, len(rewards))
            if window_size > 0:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(moving_avg, label=f'ε={epsilon}')
        
        plt.title('Récompenses moyennes (moyenne mobile)')
        plt.xlabel('Épisodes')
        plt.ylabel('Récompense moyenne')
        plt.legend()
        
        # Nombre d'étapes moyen
        plt.subplot(2, 1, 2)
        for epsilon in self.epsilon_values:
            steps = results['q_learning'][epsilon]['steps']
            window_size = min(100, len(steps))
            if window_size > 0:
                moving_avg = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
                plt.plot(moving_avg, label=f'ε={epsilon}')
        
        plt.title('Nombre d\'étapes moyen (moyenne mobile)')
        plt.xlabel('Épisodes')
        plt.ylabel('Nombre d\'étapes moyen')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Affichage des politiques
        self.display_policies(results)
        
    def display_policies(self, results):
        """
        Affiche les politiques générées par chaque algorithme.
        
        Args:
            results: Dictionnaire contenant les résultats
        """
        # Visualiser la politique de Value Iteration
        policy_vi = results['value_iteration']['policy']
        values_vi = results['value_iteration']['values']
        
        print("\nPolitique optimale de Value Iteration:")
        self.affichage.afficher_policy(policy_vi)
        self.affichage.afficher_valeurs_etat(values_vi)
        
        # Visualiser la politique de Policy Iteration
        policy_pi = results['policy_iteration']['policy']
        values_pi = results['policy_iteration']['values']
        
        print("\nPolitique optimale de Policy Iteration:")
        self.affichage.afficher_policy(policy_pi)
        self.affichage.afficher_valeurs_etat(values_pi)
        
        # Visualiser les politiques de Q-Learning
        for epsilon in self.epsilon_values:
            policy_ql = results['q_learning'][epsilon]['policy']
            values_ql = results['q_learning'][epsilon]['values']
            
            print(f"\nPolitique optimale de Q-Learning (ε={epsilon}):")
            self.affichage.afficher_policy(policy_ql)
            self.affichage.afficher_valeurs_etat(values_ql)
    
    def analyze_epsilon_impact(self, results):
        """
        Analyse l'impact du paramètre epsilon sur l'exploration et les performances.
        
        Args:
            results: Dictionnaire contenant les résultats
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Impact du paramètre epsilon sur les performances de Q-Learning', fontsize=16)
        
        # 1. Convergence des récompenses
        axes[0, 0].set_title('Convergence des récompenses')
        axes[0, 0].set_xlabel('Épisodes')
        axes[0, 0].set_ylabel('Récompense cumulative')
        
        for epsilon in self.epsilon_values:
            rewards = results['q_learning'][epsilon]['rewards']
            # Récompense cumulative
            cumulative_rewards = np.cumsum(rewards)
            axes[0, 0].plot(cumulative_rewards, label=f'ε={epsilon}')
        
        axes[0, 0].legend()
        
        # 2. Nombre d'étapes par épisode
        axes[0, 1].set_title('Nombre d\'étapes par épisode')
        axes[0, 1].set_xlabel('Épisodes')
        axes[0, 1].set_ylabel('Nombre d\'étapes')
        
        for epsilon in self.epsilon_values:
            steps = results['q_learning'][epsilon]['steps']
            window_size = min(100, len(steps))
            if window_size > 0:
                moving_avg = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
                axes[0, 1].plot(moving_avg, label=f'ε={epsilon}')
        
        axes[0, 1].legend()
        
        # 3. Récompense moyenne finale
        epsilons = self.epsilon_values
        final_rewards = [np.mean(results['q_learning'][eps]['rewards'][-100:]) for eps in epsilons]
        
        axes[1, 0].bar([f'ε={eps}' for eps in epsilons], final_rewards)
        axes[1, 0].set_title('Récompense moyenne (derniers 100 épisodes)')
        axes[1, 0].set_ylabel('Récompense moyenne')
        
        # 4. Similarité avec la politique optimale (Value Iteration)
        policy_vi = results['value_iteration']['policy']
        similarities = [self.calculate_policy_similarity(policy_vi, results['q_learning'][eps]['policy']) for eps in epsilons]
        
        axes[1, 1].bar([f'ε={eps}' for eps in epsilons], similarities)
        axes[1, 1].set_title('Similarité avec la politique optimale (Value Iteration)')
        axes[1, 1].set_ylabel('Similarité (%)')
        
        plt.tight_layout()
        plt.show()
        
        # Analyse qualitative
        print("\nAnalyse de l'impact du paramètre epsilon:")
        print("------------------------------------------")
        print("Le paramètre epsilon contrôle le compromis entre exploration (actions aléatoires) et exploitation (actions optimales connues):")
        
        for epsilon in self.epsilon_values:
            print(f"\nEpsilon = {epsilon}:")
            print(f"  - Temps d'exécution: {results['q_learning'][epsilon]['time']:.2f} secondes")
            print(f"  - Récompense moyenne (derniers 100 épisodes): {np.mean(results['q_learning'][epsilon]['rewards'][-100:]):.2f}")
            print(f"  - Similarité avec Value Iteration: {self.calculate_policy_similarity(policy_vi, results['q_learning'][epsilon]['policy']):.2f}%")
        
        # Conclusion
        print("\nConclusion sur l'impact d'epsilon:")
        print("Une valeur d'epsilon plus élevée favorise l'exploration, ce qui peut être bénéfique dans les premières étapes de l'apprentissage")
        print("mais peut limiter la convergence vers une politique optimale à long terme.")
        print("Une valeur d'epsilon plus faible favorise l'exploitation des connaissances actuelles, permettant une convergence plus rapide")
        print("mais risquant de rester bloqué dans des optima locaux si l'exploration initiale est insuffisante.")


def main():
    # Créer l'objet de comparaison
    comparison = AlgorithmsComparison(grid_size=7)
    
    # Exécuter tous les algorithmes
    results = comparison.run_all_algorithms(episodes=1000)
    
    # Afficher la comparaison
    comparison.display_comparison(results)
    
    # Analyser l'impact d'epsilon
    comparison.analyze_epsilon_impact(results)
    
    print("\nComparaison des algorithmes terminée.")


if __name__ == "__main__":
    main()