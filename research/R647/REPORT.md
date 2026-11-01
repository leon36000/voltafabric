# R647 — VFB3 refus dans le vrai `binary_loop`

## Changement

Ajout d'un test unitaire dans `src/strata_native_bridge.rs` qui appelle directement le vrai `binary_loop` avec un `Coordinator` CPU sans sessions worker et un ledger local préexistant, vide et isolé. Il envoie six requêtes VFB3 synthétiques incorrectes : identité run, modèle, plan, epoch, layer, puis champ réservé non nul.

Pour chaque cas, le test vérifie le code `INVALID_INPUT`, les octets exacts de la réponse VFR3 d'échec liée au stamp admis, zéro ligne de résultat, l'arrêt du curseur avant le corps sentinelle et l'absence de fichier dans le ledger. Les cinq divergences d'identité lisent l'en-tête et le bloc d'identité puis s'arrêtent; le champ réservé est rejeté après l'en-tête, avant la lecture de l'identité. Aucune réservation ni dispatch worker ne se produit.

## Vérification CPU

- `rustfmt --edition 2021 --check src/strata_native_bridge.rs` — PASS.
- `cargo test --manifest-path <snapshot-R643>/Cargo.toml --locked --offline --features native-ggml-cpu --lib strata_native_bridge::identity_tests::binary_loop_rejects_identity_mismatch_before_body_or_claim -- --nocapture` — 1/1 PASS, 149 tests filtrés.
- Un avertissement de compilation préexistant mentionne `CapturedQsa.input` et `.output` non lus dans `sc117_layer_chain.rs`.

Le snapshot source réduit omet huit fichiers inclus au moment de compiler d'autres modules de test. Des fichiers vides servant uniquement à satisfaire `include_bytes!` ont été créés dans ce snapshot sous `.runs`; le test R647 ne les lit pas et ils ne sont pas inclus dans la publication GitLab. Le test filtré n'exécute pas les autres tests ni les fixtures modèle manquantes.

## Réutilisation et limites

R647 durcit une frontière de VoltaFabric; il n'ajoute pas un nouveau composant externe. La réutilisation moteur déjà établie dans cette vague reste le GGML CPU épinglé de llama.cpp (R643), avec résident GGML puis Fleet/Collector VoltaFabric vérifiés séparément en R644. Ce test n'appelle ni GGML, ni checkpoint, ni Strata runtime/donneur, ni socket, ni worker, ni génération complète. Pas de GPU, RDMA, réseau de production ou mesure de performance.

## Suite

Le prochain incrément utile est le chemin positif synthétique du vrai `binary_loop` avec une session worker CPU isolée et un payload test, en conservant des identifiants frais et un ledger de test dédié. La validation modèle/checkpoint, appel Strata réel, transport inter-machine et GPU restent des admissions distinctes.
