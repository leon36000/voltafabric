# R676 — barrière d'acceptation hôte et politique de réutilisation

## Résultat

À partir de R674/R675, l'adaptateur expose `RouteGraphInvocation`, une barrière d'hôte pour un calcul GGML. Elle reçoit le nombre attendu de transactions expert avant calcul, enregistre chacune sous son identité complète, refuse les doublons, et interdit le cache du graphe dès qu'il contient une transaction VoltaFabric. Après calcul, elle n'accepte le graphe qu'une fois, uniquement si le statut GGML est SUCCESS, si le nombre attendu de transactions a été enregistré, et si chacune est READY. Une route manquante, un échec du callback dont GGML n'aurait pas propagé le statut, ou un rejeu empêche l'hôte de publier le résultat.

La fixture assemble une invocation complète avec une route prête et vérifie que la nouvelle identité reconstruit le résultat [0.75,1.25,7.5,12.5]. Elle construit aussi une invocation de deux transactions dont la deuxième reste en attente : malgré le statut GGML SUCCESS de la partie calculée, la barrière refuse la sortie du batch. Une deuxième acceptation de l'invocation complète est également refusée.

## Réutilisation du donneur

Le code de cycle d'inférence étudié dans llama.cpp MIT épinglé (`3cf03257f219afbe7334045ff7c6a06ac68c627d`) peut réutiliser un graphe GGML compatible. VoltaFabric attache au nœud une transaction identifiée à usage unique. La barrière traduit cette différence en règles concrètes pour le point hôte d'intégration : créer Stamp/transactions neufs par invocation routée, ne pas conserver le graphe de cette invocation, et valider tous les statuts avant de remettre `llm_graph_result` à l'appelant.

Cette tranche ajoute la primitive et son test à l'adaptateur isolé. Elle ne modifie pas le code llama.cpp et n'affirme pas que son `llm_graph_result` appelle déjà la barrière.

## Vérification

- Construction de la cible `vf_route_graph_dynamic_capture_test` avec C++17, `-Wall -Wextra -Werror`.
- CTest ciblé `route_graph_transaction`, `route_graph_dynamic_capture`, `route_graph_vfr3` : 3/3 pass.
- Le test de processus R673 et son reçu `RESERVED_UNKNOWN_NO_REPLAY` n'ont pas été rejoués.

## Limites et suite

Test CPU avec routes et poids synthétiques. Pas de callback `build_moe_ffn` de llama.cpp, pas de checkpoint/model réel, génération, graphe CUDA, GPU, réseau inter-PC/RDMA, registre R53 de production ni mesure de performances. Aucun changement de formats : NVFP4/QPN2 K16, MXFP4 K32 et IQ3 demeurent séparés.

R677 : appliquer l'interface dans une tranche source llama.cpp épinglée : empêcher `process_ubatch` de réutiliser le résultat si le graphe contient le callback VoltaFabric, construire une transaction fraîche, puis faire échouer le host avant publication si `RouteGraphInvocation::accepts` retourne false. Le callback MoE ne remplacera que l'agrégat routé; shared/residual et coefficient unique restent invariants.
