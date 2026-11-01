# R666 — Importer le routage Qwen2MoE épinglé dans une Session identifiée

## Source donneuse et contrat observé

Le code cible les sémantiques de `ggml-org/llama.cpp` au commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, sous licence MIT. Dans [src/llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-graph.cpp) le chemin Qwen sélectionne les experts par `ggml_argsort_top_k`, rassemble leurs probabilités et applique l'échelle des experts avant de construire la FFN MoE; [src/models/qwen2moe.cpp](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/qwen2moe.cpp) définit la FFN Qwen2MoE à gate/up/down distincts. Le manifeste et la licence du dépôt donneur restent la source d'autorité.

L'adaptateur VoltaFabric reçoit les IDs d'experts sélectionnés et les poids **finaux** du donneur, rangés suivant le tenseur GGML `[top_k, tokens]` (slot contigu). Il conserve l'ordre des slots, convertit exactement chaque `f32` vers `f64`, et ne trie pas, ne renormalise pas et n'applique pas d'échelle. Un objet de routes privé lie ensuite le `Stamp` et la tentative non nulle à l'appel composé gate/up/SwiGLU/down. L'appel conserve le contrat R665 : les sorties down restent non pondérées en FP32 jusqu'au Collector ordinaire, qui les arrondit/décode FP16 et applique le coefficient exactement une fois en ordre token/slot.

Aucun code moteur llama.cpp n'est embarqué ou exécuté comme runtime ici. R666 adapte une frontière de données inspirée de la source épinglée; les tableaux donneurs utilisés par les tests sont synthétiques et leur provenance n'est pas attestée par l'adaptateur.

## Modifications VoltaFabric

- `src/routing.rs` : importer un lot donneur borné, rejeter tentative nulle, formes incomplètes, expert négatif/inconnu/dupliqué, poids non finis ou négatifs; valider le registre et les identités avant de retourner le lot opaque.
- `src/session.rs` : exposer l'entrée FFN depuis ce lot et réutiliser la tentative donneur pour la réservation locale et les trois étapes de Session. L'ancienne entrée synthétique conserve son comportement automatique.
- `tests/native_matrix_session.rs` : vérifier l'ordre, les poids déjà mis à l'échelle sans somme-un forcée, les refus d'entrées invalides, l'identité/tentative et le chemin Q8_0 synthétique jusqu'au Collector FP16/f64.

## Vérification

Dans le harness isolé `.runs/r663-exact-collector-test-20261101/harness`, lié aux bibliothèques GGML CPU épinglées déjà construites, sans téléchargement ni GPU :

- `VF_GGML_NATIVE_BUILD=... cargo +1.90.0 test --locked --offline --features native-ggml-cpu --test native_matrix_session` : **11/11 réussis** (dont les deux nouveaux tests d'import et la FFN composée).
- Le test `qwen2moe_compound_ffn_uses_one_identity_and_returns_unweighted_down_rows_and_preflights_all_stages` en profil release, locked/offline : **1/1 réussi**.
- `rustfmt +1.90.0 --edition 2021 --check src/routing.rs` : réussi.

Les deux avertissements Rust déjà présents dans la base demeurent (`RawProjectionInputMeta` et champs `CapturedQsa` non lus). Aucun code ni fichier donneur n'a été modifié.

## Limites et suite

Ceci ne raccorde pas encore un callback au graphe/exécuteur llama.cpp : il manque le pont réel qui expose les tenseurs `selected_experts` et les poids finaux du graphe donneur à `Qwen2MoEDonorSelection`. Les tests construisent ces tenseurs à la main; ils ne reproduisent pas les ex æquo du vrai top-k. Il n'y a pas de checkpoint réel, tokenizer, attention, résiduel, boucle autorégressive, génération complète, admission partagée R53, GPU, transport inter-machine ou mesure de performance. L'essai utilise exclusivement les fixtures synthétiques Q8_0 et ne constitue pas une preuve pour QPN2/NVFP4 K16, MXFP4 K32 ou IQ3.

Prochaine tranche : créer une frontière de callback au niveau du producteur llama.cpp qui transmet les IDs et poids **après sélection et échelle** avec `Stamp`/attempt alloués une fois, puis construire et comparer le même FFN contre l'exécution indépendante du donneur CPU sur une fixture compatible, avant d'élargir vers la génération.
