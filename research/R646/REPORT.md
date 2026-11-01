# R646 — champs d'identité VFB3 réutilisés par le parseur

## Changement

`parse_vfb3_identity_fields` valide les 52 octets d'en-tête, le marqueur VFB3, les longueurs d'identité, le champ réservé et les octets concaténés run/modèle/plan. Il retourne le batch et l'attempt du même en-tête. `binary_loop` utilise maintenant ce helper avant son chemin de dispatch; le contrôle du format d'opération, des dimensions et de la politique attempt non nul reste à l'appelant.

Le harness R646 construit un en-tête et le bloc d'identité VFB3 avec des valeurs binaires, en utilisant le stamp du reçu produit par le résident GGML et le `Fleet` Rust. Il vérifie les valeurs batch/attempt, puis rejette séparément les divergences run, modèle, plan, epoch et layer, un champ réservé non nul, un en-tête raccourci, un bloc d'identité tronqué et une longueur run nulle. Les assertions R644/R645 sur la sortie FP16/f64 et l'application unique du coefficient restent dans le même test.

## Vérification

- `rustfmt --check src/strata_native_bridge.rs research/R646/cpu-harness/native_fleet_ggml.rs` — PASS.
- Harness autonome, verrouillé et hors ligne : `cargo test --manifest-path research/R646/cpu-harness/Cargo.toml --locked --offline --test native_fleet_ggml -- --nocapture` — 1/1 PASS.
- L'avertissement de compilation restant concerne deux champs inutilisés dans `sc117_layer_chain.rs`, en dehors de cette tranche.
- Une tentative de `cargo test --lib` à la racine a été bloquée avant les tests par les fixtures `include_bytes!` absentes du snapshot réduit. Le harness dépend du même crate source et évite de compiler les modules de test non liés; il ne remplace pas les preuves manquantes des autres tests.

## Portée

Le test appelle le helper commun au parseur à partir de champs encodés VFB3 et d'un reçu Fleet synthétique. Il ne lance pas `binary_loop`, ne lit/écrit pas une trame VFR3, ne réserve pas de ticket via le parseur, n'ouvre pas de socket, ne démarre aucun worker Strata et ne charge aucun checkpoint. Aucun GPU, réseau de production, RDMA ou comparaison de performance.

Le composant natif réutilisé reste le GGML CPU provenant de llama.cpp à la révision épinglée en R643. VoltaFabric garde la validation d'identité et le Collector; aucune adoption en bloc d'un autre moteur n'est revendiquée.

## Suite

La prochaine frontière décisive est une invocation contrôlée du `binary_loop` réel avec un coordinator CPU et une requête synthétique, afin de vérifier le refus VFR3 avant dispatch et l'absence de claim. Elle doit rester bornée et ne pas emprunter le checkpoint complet, le réseau inter-PC ou le GPU.
