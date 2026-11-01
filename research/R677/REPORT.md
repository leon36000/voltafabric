# R677 — résolution des propriétaires pour les routes GGML dynamiques

## Changement

La transaction GGML peut maintenant transporter `owner_by_expert`, une projection du registre qui associe chaque ID d'expert à un type de propriétaire (`-1` distant, `0..2` local). Quand le nœud GGML révèle les IDs choisis, la transaction construit `owner_kinds` dans le même ordre slot/token que les poids finaux et les transmet au Collecteur.

La validation rejette avant dispatch une table mal dimensionnée, une valeur propriétaire hors domaine, une table combinée ambiguë avec des propriétaires par route, ainsi qu'un ID d'expert dynamique hors plage. Cela permet à un futur hook du modèle de laisser la sélection top-k à GGML tout en résolvant les destinations d'après l'identité d'expert réellement choisie.

## Point d'intégration donneur

Donneur étudié : `ggml-org/llama.cpp`, commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, licence MIT. Dans `src/models/qwen2moe.cpp`, le graphe appelle `build_moe_ffn` pour la branche routée, puis traite la branche shared et le résiduel séparément. Dans `src/llama-graph.cpp`, `build_moe_ffn` produit les IDs sélectionnés, les poids finaux et le tenseur de sorties experts avant leur pondération par GGML. Ces tenseurs définissent la couture visée : fournir au hook les sorties d'experts encore non pondérées, l'entrée d'activation, les IDs sélectionnés et les coefficients définitifs. Le Collecteur VoltaFabric garde alors l'unique multiplication du coefficient. `llama_context::process_ubatch` doit encore fournir une transaction et une identité neuves et valider l'invocation avant publication.

Cette tranche modifie seulement l'adaptateur et son test. Elle ne modifie pas les sources llama.cpp et ne prétend pas que le callback Qwen2MoE appelle déjà VoltaFabric.

## Vérification

- Cible CPU `vf_route_graph_dynamic_capture_test`, construite en C++17 avec `-Wall -Wextra -Werror`.
- CTest ciblé : `route_graph_transaction`, `route_graph_dynamic_capture`, `route_graph_vfr3` : 3/3 pass.
- Le test dynamique fait correspondre les experts `[3,1,4,7]` à `[2,-1,-1,-1]` et `[2,5,6,0]` à `[1,0,2,0]` à partir d'une même table par expert.
- Une valeur propriétaire hors plage et la présence simultanée des deux formes de mapping sont refusées sans appel du Collecteur.
- Les poids finalisés restent inchangés à l'entrée du Collecteur; la fixture vérifie encore l'agrégat indépendant `[0.75,1.25,7.5,12.5]`.

## Limites et suite

Fixture CPU GGML avec poids/routes synthétiques. Pas d'intégration du callback au code Qwen2MoE, pas de checkpoint réel, génération complète, GPU/graphe CUDA, connexion inter-PC, qualification RDMA, mise à jour R53 de production ni comparaison de performances. Les formats IQ3, MXFP4 K32 et NVFP4/QPN2 K16 restent séparés.

R678 : intégrer une couture Qwen2MoE étroite au commit donneur épinglé, créer et conserver les transactions jusqu'à la fin de `process_ubatch`, couper la réutilisation quand le hook routé est actif, et faire échouer atomiquement l'hôte si chaque couche attendue n'est pas READY. Conserver shared/residual et les coefficients exactement une fois.
