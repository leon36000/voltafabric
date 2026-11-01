# R674 — contrainte d’usage unique face au graphe llama.cpp réutilisé

## Résultat

La révision épinglée `ggml-org/llama.cpp` (`3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT) peut réutiliser le même objet `ggml_cgraph` entre deux `process_ubatch` lorsque ses paramètres et entrées sont compatibles. La transaction `RouteGraphTransaction` VoltaFabric est, à dessein, à usage unique. La connecter telle quelle à un graphe gardé en cache ferait donc échouer le lot suivant, ou risquerait de garder une identité périmée si cette garde était supprimée.

Le test dynamique a été renforcé : après une première exécution acceptée, il change les logits d’entrée pour produire les routes `[2,5,6,0]` et de nouveaux poids, puis recalcule le même graphe et la même transaction. GGML calcule bien les nouvelles routes, mais la transaction rejette le deuxième lot avant un second appel au collecteur et avant le nœud aval. CTest passe 3/3.

## Vérification ciblée du donneur

- `llama_context::process_ubatch` passe par `llm_graph_result::can_reuse` ; si le graphe est gardé, il met à jour ses entrées et le recalcule (`src/llama-context.cpp:1353-1406`).
- `llm_graph_result::can_reuse` compare les paramètres puis vérifie les entrées; aucun renouvellement de transaction VoltaFabric n’existe dans cette frontière (`src/llama-graph.cpp:1406-1435`).
- `llama_context::graph_compute` soumet le graphe au scheduler asynchrone (`src/llama-context.cpp:2539`). L’état de la transaction custom doit donc être vérifié explicitement avant d’accepter le résultat du lot.
- `RouteGraphTransaction::compute` refuse une transaction dont l’état n’est plus `pending` ou dont le calcul a déjà commencé (`adapters/native_ggml/route_graph_transaction.cpp:106-110`). R668 avait déjà démontré que `cb_eval` ne propage pas à lui seul un échec du graphe jusqu’à `ggml_status`.

Sources primaires épinglées :

- [Cycle de vie et réutilisation de graphe](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-context.cpp#L1353-L1415)
- [Décision de réutilisation dans llm_graph_result](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-graph.cpp#L1406-L1435)
- [Soumission du calcul au scheduler](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-context.cpp#L2520-L2547)

## Conséquence d’intégration

Le raccord llama.cpp doit créer un nouveau Stamp et une transaction par exécution, ou empêcher la réutilisation du graphe qui porte la transaction. Le host doit ensuite inspecter le statut de toutes les routes avant de rendre `llm_graph_result` exploitable; toute route absente, UNKNOWN ou invalide doit faire échouer le lot atomiquement. Le `cb_eval` générique n’est pas une solution de remplacement.

R675 doit implémenter et tester cette frontière d’invocation côté graphe/host avant de brancher le callback Qwen2MoE. Le callback final devra remplacer uniquement l’agrégat des experts routés, laisser une seule application des coefficients donneurs, et préserver l’expert partagé et le résiduel.

## Vérification et limites

Sur AArch64 CPU : `vf_route_graph_dynamic_capture_test` recompilé avec C++17, `-Wall -Wextra -Werror`; les CTests ciblés `route_graph_transaction`, `route_graph_dynamic_capture`, `route_graph_vfr3` passent 3/3. Le test de processus R673 et son reçu `RESERVED_UNKNOWN_NO_REPLAY` n’ont pas été relancés.

Cette preuve est un test de la frontière GGML et une lecture ciblée de la source donneuse. Elle n’intègre pas encore le callback dans llama.cpp et ne charge ni checkpoint ni modèle. Aucun test GPU, graphe CUDA, inter-PC/RDMA, R53 de production, génération complète ou performance n’est qualifié.
