# R675 — transaction fraîche sur un graphe GGML reconstruit

## Résultat

R674 a montré que le graphe GGML recalcule bien des routes différentes, mais qu'une `RouteGraphTransaction` consommée refuse ce second calcul. R675 vérifie le chemin d'intégration sûr le plus simple : reconstruire le graphe avec une transaction neuve et une nouvelle identité complète pour chaque invocation qui utilise la distribution VoltaFabric.

La fixture CPU exécute un premier graphe avec les routes [3,1,4,7], puis confirme dans R674 que le rejeu avec les nouvelles routes [2,5,6,0] est refusé avant un second appel au Collector ou au nœud aval. Une invocation indépendante construit ensuite un nouveau contexte/graphe, instancie une transaction avec run/modèle/plan/couche/epoch/batch/attempt frais, et envoie les routes [2,5,6,0] et poids finaux [0.375,0.125,0.375,0.125]. Le Collector accepte une fois, produit [0.75,1.25,7.5,12.5] selon l'oracle scalaire FP64, et le nœud aval s'exécute une fois.

## Adaptation de composants

Cette tranche réutilise le nœud custom GGML et la transaction identifiée VoltaFabric, puis adopte explicitement la politique de cycle de vie imposée par le llama.cpp épinglé : toute route distribuée liée à un graphe doit avoir un Stamp/transaction neuf et le host doit valider l'état prêt avant d'exposer le résultat. Pour l'intégration initiale, un graphe neuf par invocation évite de réutiliser un `userdata` qui pointe vers une transaction consommée. Le surcoût et les possibilités de réutilisation sélective du calcul hors-routage ne sont pas mesurés.

Source donneuse : [cycle de graphe llama.cpp épinglé](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-context.cpp#L1353-L1415), licence MIT. R675 n'ajoute ni ne modifie de code llama.cpp lui-même.

## Vérification

- Hôte AArch64, cible compilée avec C++17 et `-Wall -Wextra -Werror`.
- CTest ciblé : `route_graph_transaction`, `route_graph_dynamic_capture`, `route_graph_vfr3` — 3/3 pass.
- Le test du processus coordonnateur/worker R673 et son reçu `RESERVED_UNKNOWN_NO_REPLAY` n'ont pas été rejoués.

## Limites et suite

Il s'agit d'une preuve de contrat CPU du nœud et d'une stratégie d'invocation, pas du raccord llama.cpp final. Aucun callback `build_moe_ffn` modifié, checkpoint réel, génération de tokens, modèle complet, GPU, graphe CUDA, réseau inter-PC/RDMA, R53 de production ou comparaison de performances n'est qualifié. Les formats QPN2/NVFP4 K16, MXFP4 K32 et IQ3 restent distincts.

R676 doit appliquer le contrat au raccord llama.cpp épinglé : nouvelle identité/transaction à chaque exécution routée et contrôle hôte du statut de toutes les transactions avant toute publication du résultat. Il devra remplacer uniquement l'agrégat des experts routés, en préservant shared expert/résiduel et en ne multipliant les poids qu'une fois.
