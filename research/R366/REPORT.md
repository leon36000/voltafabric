# R366 — Restauration GitLab et contrôles CPU

## Base restaurée

Clone GitLab privé de la branche R365 au commit `604764703f0a12af4fe264041ebecc0ed9c21a90` ; arbre propre au début des contrôles. Rust et Clippy épinglés : 1.90.0. Dépendances Cargo existantes en cache ; aucune installation et aucun GPU. Le clone comprend les petites fixtures présentes dans GitLab, contrairement à la reconstruction texte partielle utilisée en R365.

Empreintes SHA-256 vérifiées :
- `fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors` : `f1bdb87fe86fcafa8ae72e4154a8b4a0e66b90eaba8366a398676cb2fbe4510b`.
- `fixtures/preweighted_mxfp4_tiny.safetensors` : `f0bfbf4220713b7aae0200e15f5232d118f0abfc65d7c322a6540a837fb279a7`.
- `tests/a53_compressed_session.rs` : `6749b187c4774747196505a785a580ba1532ee432bfab0a7d32d3cc1f95b6aa2`.

## Contrôles frais

Toutes les commandes Cargo ont été exécutées hors ligne, verrouillées, avec un seul job ; les tests de session utilisaient un répertoire temporaire sous Naos.

- `cargo test --offline --locked --lib` : **106 réussis, 3 ignorés, 0 échec**. Les tests ignorés restent non exécutés.
- `cargo clippy --offline --locked --lib --test routing_heat -- -D warnings` : succès avec la version épinglée, sans exemption.
- `cargo test --offline --locked --test routing --test routing_heat` : **13 + 3 réussis**.
- `cargo test --offline --locked --test a53_compressed_session real_expert_kernel_equals_the_dense_oracle_bit_for_bit` : succès sur l'expert du petit checkpoint restauré.
- `cargo test --offline --locked --test a53_compressed_session cli_runs_a_compressed_series_end_to_end` : succès sur la série CPU locale.
- `cargo test --offline --locked --test a53_compressed_session one_transfer_then_120_exact_batches_with_two_weightless_epoch_renewals` : succès sur 120 lots de test avec poids résidents.

Ces tests n'établissent ni génération d'un modèle complet, ni exécution CUDA, ni distribution inter-PC ou RDMA, ni performance. Les tests matériels ignorés ne sont pas exécutés. La fixture experte ne prouve pas l'admission d'un modèle complet. Les anciens UNKNOWN restent consommés. La copie archivée des registres n'est pas une lease active.

## État des ressources et prochaine étape

La machine PC6 a peu de swap libre malgré sa RAM disponible ; toute expérience inter-PC doit repasser les admissions de mémoire, stockage, processus et transport avant soumission. Les actifs VoltaFabric supprimés sur PC1/PC6 ne sont pas supposés présents. Le système Strata et son modèle sont encore visibles sur PC1, sans preuve d'intégration au moteur VoltaFabric. Poursuivre par une frontière d'expert réelle compatible avec le format et l'hôte disponible, puis relier au chemin de génération ; ne pas confondre IQ3 et NVFP4/MXFP4.
