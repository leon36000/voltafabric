# R644 — GGML CPU resident vers Fleet et Collector Rust

## Changement

Ajout d'un harness Cargo autonome sous `research/R644/cpu-harness/`. Il dépend du crate VoltaFabric de la racine avec la feature `native-ggml-cpu` et teste un expert synthétique, construit selon le format de la couche 0 de l'index SC117 suivi dans le dépôt. Aucun fichier de poids ni pack modèle n'est lu ou créé.

Le test utilise `NativeLayout::parse_sc117_v3`, `Resident::load`, le vrai fournisseur C++ GGML et `Fleet::run_fp16_collected`. Il compare la sortie collectée à la conversion FP16 puis au produit f64 indépendant par le coefficient 0,25. La sortie est non nulle, donc l'assertion détecte aussi l'absence ou la double application du coefficient. Il vérifie également le refus d'un epoch incorrect avant consommation de l'essai, le refus d'un plan différent et le refus de rejouer le même stamp/batch.

Le harness existe parce que la compilation directe du snapshot réduit R642 avec `cargo test --test` tentait aussi de compiler tous ses bins; plusieurs fixtures de binaires n'étaient pas incluses dans ce snapshot. Le harness en dépendance compile seulement la bibliothèque et le test ciblé, avec son propre lockfile, sans masquer une erreur source.

## Vérification

- `rustfmt --check research/R644/cpu-harness/native_fleet_ggml.rs` — PASS.
- `VF_GGML_NATIVE_BUILD=<build GGML CPU isolé> CARGO_TARGET_DIR=<target sous .runs> cargo test --manifest-path research/R644/cpu-harness/Cargo.toml --locked --offline --test native_fleet_ggml -- --nocapture` — 1/1 PASS.
- Le build GGML/C++ et son `native_provider` CTest 1/1 proviennent de R643 et sont repris ici comme preuve donneuse, pas comptés comme un second test R644.

## Portée et suite

Cette preuve relie réellement un résident natif GGML CPU au routage VoltaFabric et au Collector Rust, mais avec un payload synthétique. Elle ne valide pas le décodage d'un tenseur du checkpoint SC117, la lecture de spans depuis le pack, le socket VFB3, le callback Strata, ni une génération complète. Les poids restent résidents pour le test, mais aucune résidence réelle de modèle n'est démontrée. Aucun GPU, réseau inter-PC, RDMA, R53 ou sortie de modèle n'a été utilisé.

Étape suivante : tester le flux d'identité VFB3 de bout en bout entre le launcher et le bridge, puis qualifier le branchement au callback Strata avec un vrai checkpoint seulement après les admissions de ressources nécessaires. Garder les UNKNOWN sans rejeu et les comparaisons de performance en attente.
