# R673 — capture des routes GGML dynamiques jusqu’au collecteur VoltaFabric

## Résultat

R673 ajoute une frontière dynamique à `RouteGraphTransaction`. Elle prend les tenseurs GGML calculés pendant le graphe — IDs sélectionnés et coefficients finaux après gating — comme dépendances de calcul, puis transmet leurs valeurs au même chemin VFB3/VFR3 et au collecteur ordinaire FP16 décodé / accumulation f64.

Le test C++ construit la route avec les opérateurs GGML épinglés : softmax, top-k, reshape des probabilités, gather des probabilités choisies et mise à l’échelle finale. Il vérifie que les IDs et poids produits par ces nœuds atteignent le collecteur, que le coefficient n’est appliqué qu’une fois et qu’un ID invalide fait échouer le graphe avant la sortie aval. Le test refuse également le rejeu du Stamp. Les trois tests CTest ciblés passent.

Le test de processus R673 fait passer une route issue de ces opérateurs à travers le client VFB3/VFR3, le vrai coordinateur natif local et son enfant `vf-rank-session`. Il a retourné PASS pour un fixture Q8_0 synthétique. Le reçu local porte `RESERVED_UNKNOWN_NO_REPLAY`; il est terminal et ne doit jamais être rejoué.

## Adaptation issue du moteur donneur

Le donneur inspecté est `ggml-org/llama.cpp` à la révision épinglée `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT). Le graphe Qwen2MoE appelle `build_moe_ffn` avec le softmax de routage et `expert_weights_scale`. Le graphe GGML produit ses IDs top-k, rassemble les probabilités sélectionnées, puis applique les normalisations et l’échelle configurées avant les projections d’experts. R672 a localisé cette frontière par lecture de source ; R673 réutilise les opérateurs/tenseurs GGML à cette frontière et fait transiter leurs valeurs dynamiques à travers l’adaptateur VoltaFabric.

Le code donneur `llama-graph.cpp` n’est pas modifié ni embarqué dans cette tranche. L’appel réel depuis `build_moe_ffn` n’est pas encore branché : l’intégration du callback/runtime Qwen2MoE reste à construire. Au raccord final, VoltaFabric devra remplacer seulement l’agrégat des experts routés et contourner la multiplication de poids donneuse qui suivrait, tout en gardant l’expert partagé et le résiduel.

## Détail de l’implémentation

- Ajoute `append_dynamic(...)`, qui attache les tenseurs d’IDs et de poids finaux au nœud personnalisé GGML sans exiger leurs données au moment de la construction du graphe.
- Vérifie les types et dimensions au montage, puis revalide au calcul les données, bornes d’IDs, unicité par token, valeurs finies et poids non négatifs avant d’appeler le collecteur.
- Lit les tenseurs avec les strides GGML `nb[]`. Le test a révélé que la sortie `ggml_argsort_top_k` n’est pas contiguë dans le cas utilisé ; une lecture plate aurait corrompu les routes.
- Une route invalide échoue atomiquement avant le collecteur et le nœud aval. La transaction ne devient pas rejouable.

## Vérification

Sur AArch64, CPU seulement : trois cibles C++17 compilées avec `-Wall -Wextra -Werror`, puis CTest ciblé 3/3 réussi (`route_graph_transaction`, `route_graph_dynamic_capture`, `route_graph_vfr3`). Le test de processus coordinateur/worker R673 a aussi retourné PASS avec identité fraîche. Voir `BUILD_TEST.txt` et le reçu local conservé sous Naos.

Le pipeline GitLab exécutera les analyses configurées de secrets, Semgrep, Trivy et le fallback du projet ; ces jobs ne remplacent pas le build C++/CTest local.

## Limites

Aucun checkpoint réel n’a été chargé ; aucun graphe d’inférence complet llama.cpp, callback/runtime Qwen2MoE, génération complète, oracle mathématique indépendant, GPU, transport inter-PC, RDMA, registre R53 de production ou comparaison de performance n’est qualifié. Le processus utilise le snapshot Rust local isolé de l’étape antérieure, pas une reconstruction complète depuis le tip GitLab courant. Le reçu local est `RESERVED_UNKNOWN_NO_REPLAY` et est conservé sans rejeu.

## Sources amont

- [Appel Qwen2MoE épinglé](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/qwen2moe.cpp#L131-L142)
- [Calcul des routes et poids dans le graphe GGML épinglé](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-graph.cpp#L2107-L2155)
- [Multiplication puis réduction des poids donneurs](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-graph.cpp#L2316-L2318)
