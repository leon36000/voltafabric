# R733 — Fenêtre KV circulaire et attention sink de DeepSeek-V4.1

## Résultat

Ajout d'une référence CPU bornée pour la tranche d'attention fenêtre du décodeur
DeepSeek-V4.1. Le cache est lié à un `Stamp` complet et à un attempt, n'accepte que
la position suivante, conserve les clés/valeurs BF16 dans l'ordre circulaire de la
source, masque les cases pas encore écrites par `-1`, puis inclut l'attention sink
dans le dénominateur softmax sans ajouter de valeur au numérateur. Les mises à jour
de cache ne sont publiées qu'après validation et calcul réussis.

Le code reprend les idées du modèle MIT épinglé; il ne copie pas son kernel. Le calcul
de référence utilise une somme stable f64 et arrondit le résultat final en BF16 RNE.
Cela vérifie la formule et l'identité des slots, sans prétendre reproduire les arrondis
du kernel TileLang (accumulation FP32, poids exponentiels BF16 et ordre de réduction).

## Frontière et intégration

`DeepSeekV41WindowCache::append_and_attend` reçoit les activations query et KV déjà
projetées, post-RoPE et BF16. Il produit une sortie par tête, étiquetée par la même
identité et la position. Les limites de cache, dimension, travail de référence, forme,
finition BF16, valeurs non finies et tentatives/positions incohérentes refusent en
échec fermé. Quatre tests couvrent l'ordre du ring et l'oracle analytique avec sink,
les positions non séquentielles, les identités divergentes, les entrées malformées et
le cas sink-only.

La tranche réalise seulement la fenêtre KV clairsemée. Les projections FP8 E4M3/E8M0,
les normalisations, RoPE, l'indexeur/top-k compressé, le groupage de sortie, HC-post et
la génération restent à raccorder. Aucune charge utile de checkpoint n'a été lue.

## Réutilisation, provenance et licence

- Donneur : `DeepSeek-V4.1-Flash`, commit MIT
  `314113f8a7dadb388215396ba5fbee2c610a04d6`.
- `model.py` SHA-256 : `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`;
  `kernel.py` SHA-256 : `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.
- Les fonctions retenues sont `get_window_topk_idxs` (`model.py`) et
  `sparse_attn`/`sparse_attn_kernel` (`kernel.py`), notamment les slots circulaires,
  sentinelles `-1`, le softmax streaming stable et la contribution sink au seul
  dénominateur. Le source est gardé comme preuve locale sous
  `evidence/continuation_a97_model_metadata/upstream_314113f8/`.
- Réutilisé dans VoltaFabric : `Stamp`, `Result`/erreurs fermées, validation BF16 RNE
  DeepSeek et les boundaries CPU identifiées en R712–R732. Aucune nouvelle dépendance.
- Aucun extrait de code donneur n'est copié; aucune qualité ou parité kernel n'est
  héritée automatiquement.

## Vérification CPU

- `cargo fmt --all -- --check` : PASS.
- `cargo test --lib --locked deepseek_v41_attention::tests -- --nocapture` : 4/4 PASS.
- Même test en `--release` : 4/4 PASS.
- `cargo clippy --locked --lib -- -D warnings` : PASS.
- Aucun GPU, réseau de production, transport inter-PC, registre R53, benchmark ou
  replay UNKNOWN n'a été utilisé.

La suite établit une primitive d'attention fenêtre identifiée; elle ne qualifie pas
un passage attention complet, une couche entière ni une génération de token.

## Suite ciblée

R734 : charger et exécuter les projections d'attention résidentes de la révision
épinglée (wq_a/q_norm/wq_b et wkv/kv_norm) au-dessus de cette fenêtre, avec une
composition qui préserve le même `Stamp` et attempt. Garder la projection de sortie
et HC-post comme frontières distinctes. Commencer par un fixture synthétique compact;
ne lire un payload réel que s'il est déjà admis et accessible, sans téléchargement.
