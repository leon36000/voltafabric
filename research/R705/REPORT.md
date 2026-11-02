# R705 — DeepSeek route vers le planificateur mémoire VoltaFabric

Date : 2026-11-02  
Branche : `r705-deepseek-gate-memory-advice-20261102`  
Parent : R704 exact-readback branch head `a9a3c13aa68cc9332d511538206c5dc376608d1d`.

## Tranche intégrée

`DeepSeekV41RouteBatch::plan_memory_advisory` appelle le planificateur générique déjà présent `planner::plan_routing_advisory_with_memory` avec le `RoutingProblem` privé et l’`attempt` stocké dans le lot. Cela relie concrètement le gate top-6 à l’admission mémoire par rang. La méthode rend un conseil de placement; elle ne charge, ne déplace et ne distribue aucun poids.

Le planificateur revalide routes, registry et dispatch, exige les profils exacts des experts routés et un envelope mémoire par rang. Les identifiants complets et les coefficients du gate passent sans recalcul. Le test vérifie leurs bits f64 dans le conseil, les budgets steady/loading, et les refus pour envelope de mauvais rang ou budget impossible. Les entrées du lot restent inchangées après succès et refus.

## Vérification CPU

- `CARGO_BUILD_JOBS=1 cargo test --offline --locked --test r702_deepseek_gate_projection` — 4 réussis ;
- `CARGO_BUILD_JOBS=1 cargo clippy --offline --locked --test r702_deepseek_gate_projection -- -D warnings` — réussi ;
- `rustfmt --edition 2021 --check src/deepseek_v41_gate.rs tests/r702_deepseek_gate_projection.rs` — réussi ;
- `git diff --check` — réussi.

Le test de liaison utilise un gate Safetensors synthétique, une topologie à un rang synthétique et six profils synthétiques. Aucun poids de checkpoint n’a été chargé et aucun worker/Session n’a été démarré.

## Réutilisation et limites

Cette tranche réutilise la frontière existante du planificateur VoltaFabric, y compris ses contrôles de budget, le dispatch validé et la conservation exacte des coefficients. Le code de gate reste l’adaptation CPU des sémantiques épinglées du donneur DeepSeek V4.1 Flash (`314113f8a7dadb388215396ba5fbee2c610a04d6`); R705 ne copie pas de nouveau code tiers. Aucun composant externe n’est supposé transférer ses garanties au moteur.

La preuve ne qualifie ni génération complète, ni checkpoint réel, ni Session, GPU ou transport inter-PC. L’inventaire R703 sur les fichiers suivis n’a trouvé aucun payload de gate réel et les experts sélectionnés 379–382 manquent. DeepSeek reste en refus atomique jusqu’à présence de tous les tenseurs, six profils, propriété résidente et admission mémoire réelle. Pas de téléchargement large, GPU, réseau de production ou rejeu UNKNOWN.
## Publication GitLab

Le code et les six fichiers R705 ont été publiés au commit `8d28f0c91d534624d192ed0a89f6c5d5c2bd6af1`; relecture exacte des sources gate/test et des quatre fichiers R705. SHA-256 source gate `6e656dcaf78df477b90b7430fc4ef06a445890a4d0d67bc360bac74161ed72f9`; test `0d416d0963ceee4c06832c0ea645d41d9def676e89988acd7e29ff7a7d4f283f`. Le pipeline 2434 est `pending` au relevé; il porte les scans configurés et le fallback, pas la compilation Rust ni le test CPU local. Aucun retry.

