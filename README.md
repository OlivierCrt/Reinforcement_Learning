### Utilisation
- Cloner le dépot
- Ajouter le dossier à vos variables d'environnements: linux: export PYTHONPATH="chemin:$PYTHONPATH"
- Executer les tests que vous voulez
---
### Particularité
J'ai fait le choix de rendre les mouvements en dehors de la grille impossibles, le reste similaire au sujet

### Comparaison

# Comparaison des algorithmes : Value Iteration, Policy Iteration et Q-Learning

## 1. Value Iteration (VI)

- **Convergence** : L'algorithme de **Value Iteration** converge généralement assez rapidement sous des conditions idéales, mais la vitesse de convergence dépend directement de la taille de l'état et du nombre de mouvements possibles. La convergence de VI est garantie dans des environnements à espace d'état fini, mais elle peut être lente si gamma est faible. Dans notre cas avec mes valeurs l'algo converge très rapidement.
  

- **Qualité de la politique finale** : Une fois que les valeurs d'état convergent, **Value Iteration** donne la politique optimale, notre environnement est déterministe donc aucuns problème. J'ai quand même rencontré quelques incertitudes quand 2 valeurs sont identiques. J'ai fais le choix dans cet algo de prendre aléatoire entre les sommets de même valeurs. La solution n est donc pas totalement unique, lpusieurs politiques optimales sont possible, avec le même cout bien sur!

  
## 2. Policy Iteration (PI)

- **Convergence** : **Policy Iteration** en théorie a tendance à converger plus rapidement que **Value Iteration** en termes de nombre d'itérations, car elle améliore directement la politique à chaque étape, plutôt que de calculer les valeurs des états à chaque fois. Cependant, elle peut être plus coûteuse par itération, car elle implique une évaluation complète de la politique à chaque étape (calcul des valeurs d'état sous la politique actuelle). Dans notre cas ça converge en une dizaine d'itérations. J'ai fais le choix de prioriser arbitrairement l'action "haut" quand égalité dans la q table, afin de montrer que avec cette implémentation 1 seule solution est possible.

- Je pense que cet algo est plus complexe en temps/ressources que VI, dans notre cas rien de choquant mais quand on augmente la taille de la grille ça devient vraiment important. J'ai fais quelques tests de mon coté avec des grilles géantes la difference est flagrante. J'aurai voulu apporter des valeurs de temps de calculs mais sur machine virtuelle time ne marche pas pour des raisons obscures.
  
- **Qualité de la politique finale** : Comme **Value Iteration**, **Policy Iteration** produit une politique optimale . La qualité de la politique finale est donc équivalente à celle de **Value Iteration**.

## 3. Q-Learning

- **Convergence** :  Sa convergence peut être plus lente que **Value Iteration** et **Policy Iteration** car l'algorithme apprend par exploration et exploitation à travers des essais et erreurs. La convergence dépend du taux d'apprentissage alpha et du facteur gamma. Ici on a un nombre fixe d'itération bien supérieur aux autres algo. Je trouve cependant cet algo plus robuste et flexible, on peut jouer sur 3 paramètres pour que l'algo soit optimiser.

## 4. L'impact du paramètre epsilon dans l'algorithme Epsilon-Greedy sur l'exploration de l'agent

Epsilon contrôle l'équilibre entre exploration (choisir une action aléatoire) et exploitation (choisir l'action avec la meilleure valeur estimée).

- **Exploration avec un epsilon élevé** (par exemple, epsilon = 1) : L'agent explore beaucoup plus, choisissant des actions au hasard de manière plus fréquente. Cela permet de découvrir plus d'options, mais peut ralentir la convergence car l'agent prend souvent des actions sous-optimales.
  
- **Exploitation avec un epsilon faible** (par exemple, epsilon = 0.1) : L'agent exploite davantage ses connaissances actuelles en choisissant souvent l'action avec la meilleure estimation. Cela peut accélérer la convergence vers la politique optimale, mais risque de rester bloqué dans un minimum local si l'exploration est insuffisante. Pour notre cas un epsilon faible 0.001 est bien plus approprié qu'un plus gros.
  
- **Stratégie** Selon moi le mieux serait de progressivement baisser epsilon en comparant les données à chaques fois, on pourrait obtenir la meileur valeur en fonction du problème.

  - **BONUS** Cet algo m'a particulièrement plus, je me suis permis de tester aussi les autres paramètres (alpha gamma) en fixant epsilon "testQlearning_gammaalpha" . Ce que j'ai pus en déduire:
      - alpha controle la vitesse d'apprentissage, de ce que j'ai compris alpha élevé = instable mais rapide, et inversement.
      - gamma joue sur le long / court termes explication: gamma élevé = l agent regarde les récompense  long terme et inversement.
      - Quleques tests de combo sont disponible dans mon code.
   


## Conclusion

- **Convergence** : **Policy Iteration** tend à converger plus rapidement en termes de nombre d'itérations, tandis que **Value Iteration** peut être plus lente en fonction du facteur de discount. **Q-Learning** peut nécessiter un plus grand nombre d'épisodes pour converger.
- **Nombre d'itérations nécessaires** : **Policy Iteration** et **Value Iteration** peuvent converger plus rapidement en termes d'itérations, tandis que **Q-Learning** nécessite généralement plus d'épisodes, mais est plus flexible dans les environnements inconnus.
- **Qualité de la politique** : Tous les algorithmes peuvent atteindre la politique optimale si suffisamment de temps et d'exploration sont accordés. **Q-Learning** est plus sensible aux choix de epsilon et au nombre d'épisodes, tandis que **Value Iteration** et **Policy Iteration** peuvent générer des politiques optimales plus rapidement dans des environnements bien définis.

Pour notre cas VI et PI sont déclarés vainqueurs!
