# R708 — vérifier les bindings avant toute lecture d'artefact MXFP4

## Changement

`run_series` fait maintenant une passe de prévalidation avant d'ouvrir un chemin d'artefact. Elle valide l'identité unique de chaque batch, les dimensions et valeurs BF16 exactes, la route, le producer/end-point, les bindings de chaque rang et la couverture complète des rangs participants. Les sources manquantes sont donc refusées avant le premier accès disque. Après cette passe, les octets chargés sont encore contrôlés contre les mêmes dispatchs MXFP4 avant toute connexion de Session.

Le test utilise le gate top‑6 synthétique `[0, 383, 382, 381, 380, 379]`, avec le seul binding de rang 0/expert 0. Le chemin d'artefact de ce rang est volontairement inexistant. Le résultat attendu est `missing MXFP4 rank endpoint`, et non l'erreur d'ouverture de fichier; aucune Session ni charge de poids n'est commencée.

## Réutilisation

R708 réutilise les validations existantes `RoutingProblem`/`RankDispatch`, `validate_binding`, `Artifact` Safetensors/MXFP4, la Session persistante et le Collector. La première passe agit comme un plan d'admission peu coûteux avant les lectures bornées. Aucun code de donneur n'est copié dans cette tranche.

## Vérification locale CPU

- Test ciblé debug : 1 réussi.
- Test ciblé release : 1 réussi.
- `cargo clippy --offline --locked --lib -- -D warnings` : réussi.
- `rustfmt --edition 2021 --check` et `git diff --check` des deux fichiers : réussis.
- Relecture GitLab des deux fichiers source : exacte.
- Pipeline #2453 en cours au moment du reçu; scans/fallback uniquement, pas compilation/test Rust.

## Limites

La route top‑6 du test est synthétique; le fichier absent est intentionnel. Cela prouve le refus des rangs/bindings manquants avant lecture, pas l'admission réussie de six experts. Le checkpoint complet et les artefacts 379–382 ne sont pas présents. Aucun worker/session, modèle, GPU, transport inter‑PC ou R53 de production n'a été utilisé; aucune génération complète n'est qualifiée.

## Traçabilité

- Branche privée `r708-mxfp4-preflight-before-artifact-io-20261102`.
- Parent R707 handoff `951e56679779cf84102acb181c0d5cc17538cc52`.
- Commit source `6c66ca3acd514f6cd6291990230f9e9d8863b404`.
- Sources/test : `src/session/mxfp4/series.rs`, `tests/r699_deepseek_memory_admission.rs`.
- Relance automatique arrêtée; aucune fusion vers `main`.
