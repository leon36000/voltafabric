# R645 — validation réutilisable de l'identité VFB3

## Tranche

Le vérificateur exact des champs run, modèle, plan, epoch et couche, déjà appelé par le parseur VFB3, est maintenant exposé comme `strata_native_bridge::identity_matches`. Sa documentation précise que le batch et l'attempt restent à valider par l'appelant avant le dispatch.

Le harness R645 réutilise l'expert synthétique GGML et le chemin natif déjà qualifié en R644. Après obtention d'un reçu `Fleet`, il vérifie l'identité correcte puis rejette séparément une mauvaise identité run, modèle, plan, epoch ou couche. Il conserve aussi les assertions R644 pour le résultat FP16 décodé, le coefficient appliqué une seule fois, la réservation d'attempt après refus d'epoch, le refus de plan et l'absence de rejeu.

## Vérification

- `rustfmt --check research/R645/cpu-harness/native_fleet_ggml.rs` — PASS.
- `VF_GGML_NATIVE_BUILD=<build GGML CPU isolé> CARGO_TARGET_DIR=<target sous .runs> cargo test --manifest-path research/R645/cpu-harness/Cargo.toml --locked --offline --test native_fleet_ggml -- --nocapture` — 1/1 PASS.
- Seul avertissement observé : les champs `CapturedQsa.input` et `.output` sont inutilisés dans un autre module du snapshot.

## Portée

Ce test exerce le même helper utilisé par le parseur, à partir d'un stamp produit par le vrai `Fleet`. Il ne construit ni ne transmet une trame VFB3, n'ouvre pas de socket, ne valide pas le batch/attempt sur le fil, n'appelle pas le callback de génération Strata et ne charge aucun checkpoint. C'est une preuve CPU ciblée, pas une preuve de génération complète.

Le donneur intégré reste llama.cpp/GGML à la révision épinglée dans R643; cette tranche réutilise le fournisseur GGML CPU et les contrats internes VoltaFabric, sans installer ni adopter un autre moteur comme cible. Aucun GPU, RDMA, réseau de production, modèle complet ou comparaison de performance n'a été exécuté.

## Suite proposée

Écrire un test in-process du parseur VFB3 réel et de sa réponse VFR3, avec une requête synthétique et le contrôle batch/attempt avant dispatch. Garder la trame/sortie atomique et l'absence de rejeu après résultat incertain; l'intégration au callback Strata demeure la prochaine frontière distincte.
