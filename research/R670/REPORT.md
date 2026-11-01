# R670 — Promouvoir le gate GGML en adaptateur réutilisable

## Changement

R670 sort l'opération de transaction de son exécutable de test et l'intègre à la bibliothèque statique `vf_native_ggml`. L'API `RouteGraphTransaction` expose trois points d'intégration : `append` ajoute l'opération personnalisée GGML au graphe, `attach_abort` lie son état au plan CPU, et `accepts` n'autorise la lecture aval qu'après `GGML_STATUS_SUCCESS` et un résultat complet.

L'objet conserve l'identité de batch run/modèle/plan/couche/epoch/batch/attempt, les routes et les coefficients tels quels. Il vérifie bornes mémoire, routes, IDs, unicité par token et finitude. Son tampon de sortie est réservé avant le calcul; le callback ne publie rien dans GGML avant le retour complet et fini. L'opération est à usage unique : un second calcul sur le même objet est refusé sans rappeler le fournisseur. Un échec est converti en `GGML_STATUS_ABORTED` entre le nœud distant et son consommateur aval.

Le donneur utilisé reste le commit MIT `ggml-org/llama.cpp` `3cf03257f219afbe7334045ff7c6a06ac68c627d`. VoltaFabric réutilise son API `ggml_custom_4d`, son `ggml_cplan.abort_callback` et son graphe CPU; aucun code GGML donneur n'est modifié.

## Vérification

Sur CPU AArch64 contre le checkout donneur exact :

- `vf_native_ggml` compile avec `-Wall -Wextra -Werror`;
- le test lié à la bibliothèque exécute le cas nominal, l'échec partiel et la tentative de rejeu;
- CTest ciblé `route_graph_transaction` : 1/1 PASS;
- le cas d'échec retourne `GGML_STATUS_ABORTED` et le consommateur aval n'est pas appelé;
- le second calcul sur la même transaction est refusé sans second appel au collecteur.

## Limites

Le callback de collecte reste un substitut CPU synthétique; le nouveau composant n'est pas encore branché à `vf-rank-session`, au Collector Rust réel ou au runtime llama.cpp. Aucun GGUF/checkpoint, génération réelle, GPU, transport inter-PC/RDMA, admission R53 ou mesure de performance n'a été utilisé. La bibliothèque et son test établissent une frontière réutilisable et fail-closed, pas la génération VoltaFabric complète.

Prochaine étape : adapter le callback à une réponse VFR3 identifiée de Session Rust, puis tester résultat complet, échec explicite, identité périmée et refus avant le consommateur aval.
