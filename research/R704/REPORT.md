# R704 — Liaison du gate DeepSeek vers le routage VoltaFabric

Date : 2026-11-02  
Branche : `r704-deepseek-gate-route-identity-20261102`  
Parent : reçu R703 `bcaa7ee42752abfa597d1333fd1e20f9021d7a3c`

## Tranche intégrée

`DeepSeekV41Gate::prepare_route_batch` convertit une décision top-6 d’un token en
`RoutingProblem` VoltaFabric et appelle le validateur commun avant de rendre le lot.
Le nouveau `DeepSeekV41RouteBatch` garde le problème privé et transporte l’attempt
non nul avec l’identité run/modèle/plan/couche/batch/epoch du `Stamp`. La méthode
refuse un attempt nul, une révision ou une couche différente du gate chargé, une
route qui n’est pas top-6 et toute incohérence registry/rank/local/order détectée
par `routing::validate`.

Les coefficients retournés par le gate sont transmis tels quels. Aucun tri,
renormalisation ou second `route_scale` n’est appliqué. Le lot peut être fourni au
planner puis à une Session par l’appelant, après la validation de mémoire et la
présence effective des poids.

## Vérification CPU

Le test de projection R702 a été étendu pour construire un lot à partir du fixture
synthétique Safetensors, vérifier l’identité complète, l’attempt 29, l’ordre des six
experts et la somme des coefficients `ROUTE_SCALE`, puis vérifier le refus du mauvais
layer et de l’attempt zéro.

- `CARGO_BUILD_JOBS=1 cargo test --offline --locked --test r702_deepseek_gate_projection` — 3 réussis ;
- `CARGO_BUILD_JOBS=1 cargo clippy --offline --locked --test r702_deepseek_gate_projection -- -D warnings` — réussi ;
- `rustfmt --edition 2021 --check src/deepseek_v41_gate.rs tests/r702_deepseek_gate_projection.rs` — réussi ;
- `git diff --check` — réussi.

## Relecture de publication

La première relecture du commit source `83d8ddf0a3b2d45ae0c490263aa0c8d84dee531e` a révélé un doublon accidentel de la fonction finale `softplus_fp32`. Le commit correctif `8b6519ba41f7a4f872612541a32ce8f05e4220a3` a retiré uniquement ce suffixe; la relecture GitLab correspond ensuite exactement au fichier testé local, SHA-256 `644e17ed6fd0cac8a13e906a413869bca3a5abe18404fe45a57d686d1fb8a4b1`. Les tests CPU et Clippy ont été relancés après cette correction et passent.

Le pipeline 2417 (R703) est `success` (scans configurés et fallback uniquement). Le pipeline 2421, associé au premier commit R704 contenant le suffixe dupliqué, est encore `running`; le pipeline 2426 du commit corrigé est `canceled`; le pipeline 2427 du reçu `02ba5eb8c91cfc8118d1ab3a6aecfdacac80003f` est `pending`. La cause exacte de l’annulation 2426 n’est pas établie. Aucun retry n’a été lancé. Ces pipelines ne compilent pas Rust et ne remplacent pas les tests CPU locaux.

## Réutilisation et limites

Cette tranche réutilise l’API `RoutingProblem` et `routing::validate` de VoltaFabric,
et s’appuie sur l’adaptation du gate DeepSeek V4.1 Flash MIT à la révision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. Aucun code donneur supplémentaire n’a
été copié. Le parser Safetensors reste la dépendance Apache-2.0 déjà épinglée.

Le test reste synthétique et CPU. R703 a parcouru 7 920 entrées GitLab et établi que
les poids réels du gate et les experts 379–382 sont absents des artefacts suivis. Ce
raccord ne lance pas le planner, ne démarre pas de Session et ne qualifie pas la
génération. Le dispatch doit rester refusé tant que le gate réel, les six profils,
leurs poids résidents et l’admission mémoire complète ne sont pas disponibles.
Pas de téléchargement du checkpoint entier, GPU, réseau de production, rejeu
UNKNOWN ou relance automatique.
