# R669 — Frontière transactionnelle d'opération GGML

## Adaptation

R669 transforme le constat R668 en une tranche C++ testable autour du GGML épinglé de `ggml-org/llama.cpp` (`3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT). L'opération `ggml_custom_4d` reçoit un tenseur CPU de sorties expertes organisées `[hidden, slot, token]` et un batch identifié (run, modèle, plan, couche, epoch, batch, attempt), avec IDs et coefficients de route inchangés.

Le callback de collecte reçoit ces sorties et écrit dans un tampon déjà réservé. L'opération valide dimensions, identité, routes uniques par token, IDs, coefficients et valeurs finies. Elle ne copie vers le tenseur GGML aval qu'après un statut complet et un résultat entièrement fini. Un échec après une écriture partielle dans le tampon est transmis au `ggml_cplan.abort_callback`; `ggml_graph_compute` renvoie alors `GGML_STATUS_ABORTED` après ce nœud, avant l'opération aval. Le résultat du graphe n'est accepté que si le statut est `GGML_STATUS_SUCCESS` et que le nœud aval a été exécuté.

La tranche réutilise les opérations personnalisées GGML et le plan/abandon CPU du donneur. Elle adapte le contrat d'abandon; elle ne reprend aucun code de calcul propriétaire du donneur.

## Vérification

Contre le checkout donneur local intact à la révision épinglée, sur CPU AArch64 :

- compilation de `vf_route_graph_transaction_test` avec `-Wall -Wextra -Werror` : réussie;
- CTest ciblé `route_graph_transaction` : 1/1 PASS;
- cas de succès : deux tokens, deux slots, poids utilisés une fois, sortie aval attendue `[2.5, 3.5, 18, 28]`;
- cas d'échec injecté : le callback écrit un élément puis échoue; le graphe retourne `GGML_STATUS_ABORTED` et le callback aval n'est jamais appelé.

Sources donneuses épinglées : `ggml/include/ggml.h` (`ggml_custom_4d`), `ggml/include/ggml-cpu.h` (`ggml_cplan`, `ggml_graph_compute`) et `ggml/src/ggml-cpu/ggml-cpu.c` (propagation de l'abandon entre nœuds).

## Limites et prochaine intégration

Le callback du test est un faux collecteur CPU déterministe. Il n'appelle ni le Collector Rust ni `vf-rank-session`; aucun GGUF/checkpoint, graphe llama.cpp complet, décodage réel, GPU, réseau, réservation R53 ou mesure de performance n'a été exécuté. Le tampon et l'état d'identité sont un adaptateur de frontière, pas encore une intégration de runtime. La prochaine tranche doit relier ce callback au résultat identifié du Session Rust et vérifier le refus d'un résultat manquant ou d'une identité périmée avant que le nœud de décodage puisse s'exécuter.
