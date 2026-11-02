# R707 — prévalidation atomique des experts MXFP4 avant session

## Changement

La validation de série MXFP4 est maintenant isolée dans `preflight_execution`. Elle vérifie toutes les identités de batch, les entrées BF16 exactes, les dispatchs de route, les bindings et les budgets d'entrée pour tous les lots avant le premier `Session::connect` ou `LOAD`. Le lancement ne commence qu'après réussite de cette passe complète.

Un test exerce le gate top‑6 DeepSeek `[0, 383, 382, 381, 380, 379]` réparti sur six rangs et fournit seulement l'artefact MXFP4 réel, épinglé MIT, de l'expert 0. La série est refusée avec `missing MXFP4 rank endpoint` avant de tenter la connexion. L'exécutable factice est volontairement introuvable; une erreur de connexion aurait produit un rapport de Session plutôt que cette erreur de prévalidation.

## Réutilisation

R707 garde le parseur/résident `mxfp4::Artifact`, `routing::dispatch`, la Session persistante et le Collector FP32 prépondéré VoltaFabric. Il rend explicite leur barrière d'admission commune. Aucun nouveau code donneur n'est copié dans cette tranche; la licence des papiers arXiv n'est pas une licence de code.

## Vérification

- Test ciblé en debug : 1 réussi.
- Même test ciblé en release : 1 réussi.
- `cargo clippy --offline --locked --lib -- -D warnings` : réussi.
- `rustfmt --edition 2021 --check` et `git diff --check` sur les deux fichiers : réussis.
- GitLab pipeline #2448 : en cours au moment du rapport; son rôle est scans/fallback configurés, pas la compilation Rust ni les tests CPU.
- Les deux fichiers source sont publiés sur la branche privée et leur relecture GitLab correspond exactement.

## Portée

Cette preuve qualifie seulement le refus atomique d'un lot top‑6 incomplet. Elle ne qualifie pas l'admission complète des six artefacts, le checkpoint complet, une génération, un modèle chargé, un GPU, le transport inter‑PC ou R53 de production. Les artefacts des experts DeepSeek 379–382 et les vrais octets du gate restent absents des sources disponibles; aucune session n'a démarré.

## Traçabilité

- Branche privée : `r707-deepseek-six-expert-atomic-admission-20261102`.
- Commit source : `e5a040b2a785b5c9f2efb0aeae1c8aa997e91cc4`.
- Parent : `6a305860030a67096f3841a7ef41828e697dc1d4`.
- Fichiers : `src/session/mxfp4/series.rs`, `tests/r699_deepseek_memory_admission.rs`.
- La relance automatique reste arrêtée; aucune fusion vers `main`.
