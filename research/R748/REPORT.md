# R748 — tester le crate Rust racine sur le runner PC4

## Pourquoi

La réussite R747 #2873 valide le harnais isolé de R746, mais son propre `Cargo.toml` redéclare les modules testés. Elle ne prouvait pas que le crate `voltafabric` compile les modules réels de `src/` avec `deepseek-v41-stream`. R748 ajoute donc tests et Clippy de la bibliothèque racine, en conservant le harnais isolé et les scans GitLab.

## Résultat de la première exécution racine

La pipeline GitLab PC4 #2878, sur le commit `18964df7f7b13d1416cbc2c3d4cfb7f1034f4154`, a terminé en échec dans le job `r746_cpu_contract` (#6710). Le job a compilé les dépendances puis échoué à la compilation des tests unitaires du crate racine : `src/session/mxfp4.rs:1489` construisait `preweighted::Completion` sans le champ `attempt`. Tests, Clippy et rustfmt n'ont donc pas été atteints.

Le diagnostic est confirmé par l'historique du fichier : R689 a ajouté le champ optionnel et son initialisation `attempt: None`; la refactorisation R714 de la frontière multi-rang a retiré cette initialisation. Le collecteur MXFP4 R9 utilisé ici est `Collector::new(problem)`, dont le contrat attend `attempt = None`; la Session vérifie séparément l'essai non nul de la requête et celui de chaque événement/réponse avant d'accepter les lignes. Le correctif restaure `attempt: None` dans l'initialiseur concerné, sans changer le protocole ni la route.

## Scans et prochaine vérification

Les trois jobs `.pre` de #2878 ont réussi : Gitleaks (#6707), Semgrep (#6708) et Trivy (#6709). Cela n'établit pas de résultat de test racine. Semgrep a déclaré 155 findings sur 6 898 fichiers, encore non triés. La CI doit être relancée par le nouveau commit correctif avant de déclarer la compilation, les tests, Clippy ou le formatage verts.

## SonarQube et GitGuardian

Aucun connecteur SonarQube/GitGuardian n'est disponible. Aucun binaire Sonar n'a été trouvé sur PC6; les contrôles PC4 antérieurs n'ont trouvé ni service local sur le port 9000 ni `SONAR_HOST_URL`. La valeur masquée de `SONAR_TOKEN` n'a pas été lue. Gitleaks ne remplace pas GitGuardian; aucun scan SonarQube ou GitGuardian n'a été exécuté.

## Limites

Cette correction CI compile du code moteur réel, mais ne relie toujours pas le sampler de génération à `Session::Reply`, ne lit aucun payload de checkpoint réel et ne qualifie ni génération complète, ni GPU, ni liaison inter-PC, ni performance.
