# R649 — deux batches VFB3 par une session CPU native

## Résultat qualifié

Le test `strata_native_bridge::identity_tests::binary_loop_runs_two_synthetic_native_cpu_batches_and_closes` appelle le `binary_loop` de production avec un processus persistant local `vf-rank-session`, rank 0, un résident GGML natif synthétique (expert 8, couche 0), puis deux requêtes VFB3 valides dans la même conversation. Elles ont le même run/modèle/plan/epoch/couche mais des batches et attempts distincts (`9649001/14`, `9649002/15`) et deux vecteurs d'entrée distincts.

Le test a comparé chaque trame VFR3 complète bit à bit à `Resident::forward` du même fournisseur, a vérifié l'ordre et les identités de réponse, puis la trame d'arrêt. Le template contient un poids de route 0,25; cette frontière retourne les résultats bruts non pondérés et ne multiplie pas une seconde fois. Le ledger propre contient exactement deux reçus `RESERVED_UNKNOWN_NO_REPLAY`, un par identité de batch/attempt. Ces reçus sont publiés sous `receipts/clean-*` et ne doivent jamais être rejoués.

## Incident de traçabilité et confinement

Un premier essai de cette tranche, avant le contrôle des reçus R648, a repris par erreur le run/batch/attempt déjà consommé par R648 (`r648-positive-binary-loop`, batch `9648001`, attempt `4`) dans un autre répertoire ledger. Cette identité UNKNOWN a donc été réservée une seconde fois par inadvertance; la transaction est hors preuve et ne justifie aucun rejeu. Le même essai a également réservé un second reçu (`9648002/5`), lui aussi mis en quarantaine avec le reste de cette tentative. Aucun de ces deux reçus n'a été rejoué après détection. Les deux reçus R648 et les deux reçus de cette tentative sont conservés sous `receipts/` pour que l'incident reste visible.

Le test final accepté utilise un nouveau run (`r649-multi-batch-binary-loop`), deux nouveaux batches et attempts, ainsi qu'un ledger frais distinct. Aucune nouvelle opération ne réutilise les identités quarantinées.

## Vérification et limites

- `rustfmt --edition 2021 --check src/strata_native_bridge.rs` — PASS.
- `cargo test --locked --offline --features native-ggml-cpu --lib strata_native_bridge::identity_tests::binary_loop_runs_two_synthetic_native_cpu_batches_and_closes -- --exact --nocapture` — 1/1 PASS, 150 tests filtrés.
- Le test compare à `Resident::forward` avec le même fournisseur; ce n'est pas un oracle mathématique indépendant. Données synthétiques uniquement.
- Un rank/session CPU; aucun checkpoint, binaire donneur Strata ou callback de génération complète chargé. Pas de GPU, RDMA, inter-PC, R53 de production ou comparaison de performance.
- Les huit stubs de compilation du snapshot réduit restent locaux sous `.runs` et ne sont pas inclus dans cette branche.

Cette preuve fait avancer le raccord natif GGML CPU vers le parseur de lots VFB3 réel et la réponse VFR3 en session persistante. Elle ne qualifie pas l'adoption complète de llama.cpp ni celle de vLLM, SGLang, TensorRT-LLM, Maya ou DeepInfer. La prochaine frontière utile est le lancement réel du pont par l'adaptateur Strata local, sans charger de checkpoint complet.
