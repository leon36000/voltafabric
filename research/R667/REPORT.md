# R667 — Adapter les strides du top-k GGML au routage Qwen2MoE

## Source donneuse et résultat

La tranche cible le commit MIT épinglé `ggml-org/llama.cpp` `3cf03257f219afbe7334045ff7c6a06ac68c627d`, déjà construit en CPU dans le lot R655. Elle réutilise ses opérateurs GGML natifs `ggml_soft_max`, `ggml_argsort_top_k`, `ggml_get_rows` et `ggml_scale`; elle n'intègre pas le runtime llama.cpp.

La lecture de `ggml/src/ggml.c` montre que `ggml_argsort_top_k` retourne une vue de l'argsort complet : l'extent `ne[0]` devient `k`, tandis que le stride `nb[1]` garde le pas de ligne du tenseur complet. L'opérateur CPU dans `ggml/src/ggml-cpu/ops.cpp` écrit une ligne par token à ce pas. Lire `k * tokens` entiers comme un tableau contigu mélangeait la fin de la première ligne avec la suivante. Le cas 3 experts × 2 tokens produisait ainsi `[0, 2, 1, 1]`; l'adaptateur copie maintenant chaque ligne avec le pas `experts`, puis expose les slots tassés `[slot, token]` à l'adaptateur R666. Le test vérifie aussi les poids softmax après l'échelle `expert_weights_scale=0.5`.

## Changements VoltaFabric

- `crates/ggml-native-ffi/src/lib.rs` : fixture bornée et CPU qui appelle les opérateurs GGML épinglés, et copie le résultat top-k en respectant le stride du view GGML.
- `tests/native_matrix_session.rs` : test à 3 experts et 2 tokens qui vérifie l'ordre par token, les poids finaux, puis importe les routes dans l'adaptateur de Session R666 sans renormalisation.

Cette frontière réutilise réellement les opérateurs GGML du donneur dans une fixture CPU; elle ne récupère pas encore les tenseurs depuis l'exécution du modèle. `ggml_get_rows` renvoie les poids finaux des slots et `ggml_scale` applique l'échelle de Qwen avant import.

## Vérification

Harness isolé `.runs/r663-exact-collector-test-20261101/harness`, bibliothèques statiques CPU GGML construites et épinglées, commande verrouillée hors ligne :

- test `native_matrix_session` en debug : **12 réussis, 0 échec**;
- même test en release : **12 réussis, 0 échec**;
- `cargo clippy` du target de test : terminé avec succès, en signalant des avertissements existants dans les modules Session/Strata;
- `rustfmt +1.90.0 --edition 2021 --check` sur les deux blocs R667 isolés : réussi. Le contrôle des fichiers entiers reste non passant à cause de sections antérieures R666 non formatées; aucun reformatage global n'a été appliqué.

Une première commande depuis le manifeste incomplet `source/Cargo.toml` a échoué avant d'atteindre ce test parce que le snapshot ne contient pas plusieurs fixtures embarquées par des cibles historiques. La preuve positive est celle du harness isolé, qui ne compile que les cibles nécessaires.

## Limites

Il s'agit d'une fixture de routage synthétique CPU, pas d'un callback du graphe llama.cpp ni d'une génération avec checkpoint. Aucun ex æquo top-k, filtre de groupes, shared expert, résiduel, tokenizer, génération autorégressive, GPU, réseau/RDMA, admission externe R53 ou mesure de performance n'est qualifié. Les formats QPN2/NVFP4 K16, MXFP4 K32 et IQ3 ne sont pas évalués.

Prochaine tranche proposée : trouver un point de capture runtime après le calcul du graphe Qwen2MoE qui peut lire les IDs et poids calculés du même batch, puis transmettre ce résultat et son identité à VoltaFabric. Le callback de construction de graphe documenté par R666 n'est pas ce point runtime.
