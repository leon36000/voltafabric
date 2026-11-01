# R648 — batch natif synthétique via `binary_loop`

## Changement

Le test appelle le vrai `binary_loop` avec un `Coordinator` mono-rang et un processus local `vf-rank-session`. Le résident reçoit un blob SC117 au format natif GGML synthétique, puis est scellé avant l'envoi d'une trame VFB3 valide. Après un batch, une trame VFB3 d'arrêt ferme proprement la session.

La réponse VFR3 complète (en-tête d'identité, position et 2560 valeurs FP32) est comparée bit à bit à `Resident::forward` pour le même vecteur d'entrée. Le routage du template porte un coefficient 0,25, mais cette frontière Strata non pondérée retourne les valeurs brutes, comme attendu avant la pondération du donneur. Le ledger local a exactement un fichier `RESERVED_UNKNOWN_NO_REPLAY` pour le run, batch et attempt de ce test. La fermeture produit sa réponse VFR3 et termine le worker.

Un premier essai de démarrage avec `vf-rank-worker` ponctuel a échoué au HELLO `Session::connect`; le code de ce binaire attend une requête de rang unique. L'échec s'est produit avant la création du `Coordinator` et avant tout ticket; le répertoire ledger était encore vide. Le chemin a été corrigé vers `vf-rank-session`, qui implémente le HELLO persistant attendu, puis le test a passé une fois. Aucun batch inconnu n'a été rejoué.

## Vérification CPU

- `rustfmt --edition 2021 --check src/strata_native_bridge.rs` — PASS.
- `cargo build --locked --offline --features native-ggml-cpu --bin vf-rank-session` — PASS.
- Test ciblé verrouillé/hors ligne `strata_native_bridge::identity_tests::binary_loop_runs_one_synthetic_native_cpu_batch_and_closes` — 1/1 PASS, 150 tests filtrés.
- L'avertissement de compilation préexistant mentionne `CapturedQsa.input` et `.output` inutilisés dans `sc117_layer_chain.rs`.

Le snapshot réduit omet huit fichiers inclus lors de la compilation des autres modules de test. Les huit stubs vides locaux utilisés pour franchir la compilation sont sous `.runs` du snapshot; le test ne les lit pas et ils ne sont pas publiés. Aucun test de ces fixtures manquantes n'est revendiqué.

## Réutilisation et portée

Cette tranche vérifie concrètement le GGML CPU épinglé par R643 dans un processus résident, puis le parseur de trame VFB3 et la réponse VFR3 de VoltaFabric. Le contrôle de sortie compare avec `Resident::forward` utilisant le même fournisseur GGML; ce n'est pas un oracle mathématique indépendant. La charge est synthétique et aucun checkpoint ou donneur Strata actif n'est chargé.

La preuve est locale CPU : une session/rang, un expert, un batch et une réservation. Elle ne qualifie ni la génération complète, ni le runtime donneur, ni les 48 sessions, ni le réseau inter-PC, la production R53, GPU, RDMA ou les performances. Aucune adoption complète de llama.cpp ou d'un autre moteur n'est revendiquée.

## Suite

R649 peut vérifier plusieurs entrées d'une même conversation locale avec des identités fraîches et des lots atomiques, puis avancer vers le raccord donneur/launcher sans charger de checkpoint complet. Un ledger ayant déjà une réservation reste consommé et ne doit pas être rejoué. Le GPU et le réseau de production restent suspendus sans admissions fraîches.
