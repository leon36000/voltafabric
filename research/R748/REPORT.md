# R748 — faire compiler et tester le crate VoltaFabric dans GitLab PC4

## Pourquoi

La réussite R747 #2873 valide le harnais isolé de R746, mais son `Cargo.toml` local redéclare les modules testés. Il ne prouvait pas que le crate `voltafabric` compile les modules réels de `src/` avec le feature `deepseek-v41-stream`. R748 ajoute donc le test et Clippy de la bibliothèque racine, tout en gardant le harnais CPU indépendant et les scanners globaux GitLab.

## Commandes sur le runner PC4

La pipeline doit exécuter sous Rust 1.90.0 depuis le registre public ECR déjà qualifié :

1. `cargo test --locked --lib --features deepseek-v41-stream`
2. `cargo clippy --locked --lib --features deepseek-v41-stream`
3. Les tests et Clippy du harnais R746 isolé
4. `rustfmt --check` sur les modules DeepSeek et le harnais

Le root lib test compile la source VoltaFabric réelle, dont le reducer d'argmax segmenté et le décodeur DeepSeek. Il n'installe pas de modèle, ne lit aucun checkpoint et ne lance aucun GPU.

## CI PC4

R748 est en attente de son premier résultat. R747 #2873 demeure la preuve établie : PC4 a exécuté le harnais isolé, avec 22 tests réussis, 0 échec et 3 ignorés faute du tokenizer externe épinglé. Ne pas étendre ce résultat au crate racine avant la nouvelle pipeline.

## SonarQube et GitGuardian

Le gestionnaire de plugins retourne zéro connecteur SonarQube/GitGuardian. Aucun binaire `sonar` ou `sonar-scanner` n'est installé sur PC6; PC4 n'a ni Docker, Podman ni Nerdctl visibles, et aucun processus n'écoute sur le port 9000. La requête locale PC4 `/api/system/status` a échoué par connexion refusée. GitLab expose seulement le nom d'une variable `SONAR_TOKEN` masquée, sans `SONAR_HOST_URL`; sa valeur n'a pas été lue. Gitleaks, Semgrep et Trivy sont injectés dans les pipelines, mais Gitleaks ne remplace pas GitGuardian et aucun scan SonarQube n'a été effectué.

Le service local n'est pas disponible par les accès inspectés. La skill dédiée demande une autorisation distincte avant d'installer/configurer SonarQube CLI et son serveur MCP; je ne fais aucun changement de configuration ni démarrage de service dans ce lot.

## Limites et suite

Cette tranche ferme un écart de test CI. Elle ne raccorde pas encore l'échange R746 à `Session::Reply`, ne qualifie aucune génération complète ou liaison inter-machine et ne représente pas une validation GPU. Une fois la pipeline lue, ajouter son reçu, puis choisir une seule intégration source testable.
