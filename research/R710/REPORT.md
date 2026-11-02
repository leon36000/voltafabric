# R710 — lecture sélective MXFP4 liée à la route et au rang

## Tranche intégrée

`read_pinned_mxfp4_rank_expert` relie une route VoltaFabric validée, son attempt non nul, un expert sélectionné et son binding `(expert, rank, local)` à un plan MXFP4 R709 puis au lecteur existant `checkpoint::read_named_tensors`. La fonction traite un expert par appel. Elle vérifie le modèle, le top‑6, la couche, le binding résident, l'index épinglé, le nom de shard, le hash et la taille du header, la longueur du fichier, les six formes/types et les bornes; elle exige exactement six SHA‑256 de payload avant la première lecture de poids.

Le lecteur existant est réutilisé pour prévalider les six tenseurs, lire seulement leurs plages et contrôler chaque digest. Toute erreur retourne une erreur sans import partiel. Le header est vérifié avant et après la lecture; la source doit rester quiescente. Le reçu conserve run/modèle/plan/couche/batch/epoch, attempt, route top‑6, expert, rang et emplacement local. Il ne crée pas encore `Mxfp4Expert`, ne démarre aucun Session/worker et ne change ni les coefficients ni l'ordre de la route.

## Preuve CPU

Le test utilise l'index et l'en-tête officiels épinglés DeepSeek‑V4.1‑Flash (révision `314113f8a7dadb388215396ba5fbee2c610a04d6`) et un shard sparse de taille logique 7 389 759 032 octets. Seul le payload synthétique d'un expert est matérialisé en mémoire. Le test montre que les six lectures coïncident exactement avec les six plages du plan et que le lecteur ne lit aucune autre plage. Des tests refusent un digest absent et un expert hors route avant toute lecture de poids, puis refusent un hash incorrect sans retourner les tenseurs précédemment lus.

- `cargo test --offline --locked --lib a148_pinned_expert_selective_read_tests` : 3 réussis.
- Même test en `--release` : 3 réussis.
- `cargo clippy --offline --locked --lib -- -D warnings` : réussi.
- `rustfmt --edition 2021 --check src/deepseek_v41.rs` et `git diff --check` : réussis.

## Limites et intégrité

Les octets de poids du modèle réel n'ont pas été lus. Les SHA‑256 des payloads viennent obligatoirement d'un manifeste indépendant fourni par l'appelant; l'index et le header épinglés n'authentifient pas les valeurs des poids. Le test sparse prouve la sélection des offsets et le refus par hash avec des payloads synthétiques, pas la présence, la résidence ou l'exactitude des experts 379–382. La fixture route `[0, 383, 382, 381, 380, 379]` reste synthétique. Aucune génération complète, aucun worker/session, GPU, transport inter‑PC, R53 de production ni UNKNOWN rejoué.

## Réutilisation et suite

Réutilisé : le lecteur borné `checkpoint::read_named_tensors`, les types Safetensors existants et le plan exact R709. Aucun code de moteur tiers n'est copié dans cette tranche; DeepSeek sert de référence de format sous MIT à la révision épinglée. Prochaine tranche : construire un owner résident par rang à partir des six tenseurs vérifiés sans matérialiser un shard, puis tester le transfert atomique vers le contrat de Session existant. Le dispatch demeure refusé tant qu'il manque un des six experts, ses digests authentifiés ou son admission mémoire.

## Publication GitLab privé

La source, le rapport, le journal de tests, le reçu et SHA256SUMS sont relus exactement sur la branche `r710-deepseek-route-expert-selective-read-20261102` au commit `ad73ca3174a4a87b42e82e5f9ed8fe00eea43fe1`; les cinq fichiers correspondent localement. Le premier commit `965fb660997ff1ad2848d25e64bf79f39efa96ce` avait stocké leurs chaînes Base64 comme texte; le commit correctif `ad73ca3174a4a87b42e82e5f9ed8fe00eea43fe1` a rétabli le texte source exact sans réécrire l’histoire. Le pipeline GitLab 2463 est en cours et ne remplace pas les tests Rust locaux.
