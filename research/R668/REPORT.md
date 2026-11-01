# R668 — Rejeter le callback d'évaluation llama.cpp comme frontière de dispatch

## Source et constat

Revue ciblée du commit MIT épinglé `ggml-org/llama.cpp` `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

L'API GGML documente `ggml_backend_sched_eval_callback` comme observation des tenseurs d'un graphe en exécution : le scheduler demande d'abord si le noeud doit être observé, puis lui transmet son tenseur après calcul et synchronisation du backend. C'est un véritable point d'observation runtime, distinct du callback `llm_graph_context::cb` de construction du graphe décrit par R666.

Ce callback ne constitue toutefois pas une frontière sûre pour remplacer les experts llama.cpp par VoltaFabric. Dans `ggml_backend_sched_graph_compute_async`, un retour `false` après observation quitte la boucle des noeuds; la fonction retourne ensuite `GGML_STATUS_SUCCESS`. `llama_context::process_ubatch` traite ce statut comme une réussite et retourne le résultat du graphe. Une tentative d'arrêter le graphe après capture des routes pour effectuer le FFN distant pourrait donc exposer une exécution partielle comme réussie, en conflit avec le refus atomique VoltaFabric. Le callback provoque aussi une synchronisation backend à chaque noeud observé.

Sources primaires épinglées :

- [contrat eval callback](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/include/ggml-backend.h#L309-L316)
- [boucle scheduler, arrêt et statut final](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-backend.cpp#L1798-L1845)
- [traitement du statut par llama.cpp](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-context.cpp#L1406-L1415)

## Décision d'intégration

Ne pas utiliser `cb_eval` comme barrière d'attente/routage ou mécanisme d'abandon pour un FFN VoltaFabric. Il reste une possibilité d'observation diagnostique où le graphe donneur continue son exécution, mais cela ne distribue pas le calcul et ne donne pas un résultat distant au graphe.

Pour une génération distribuée sûre, il faut une opération d'intégration qui consomme les sorties VoltaFabric comme entrée de la suite du graphe et propage une erreur explicite si une route, un propriétaire ou un résultat manque. Une modification du donneur devrait inclure un statut d'abandon non-succès et une politique d'état/KV qui exclut les batches incomplets; ces garanties ne sont pas apportées par le callback actuel.

## Portée

Analyse source seule, sans changement ni compilation du checkout donneur, sans modèle ou checkpoint, sans exécution GPU, réseau/RDMA, admission R53 ou mesure de performance. Ce résultat élimine une fausse frontière d'intégration; il ne qualifie pas une génération llama.cpp/VoltaFabric.
