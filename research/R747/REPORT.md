# R747 — exécuter le contrat CPU dans le runner GitLab PC4

## Changement CI

Le projet privé n’avait pas de `.gitlab-ci.yml` à la racine. Les pipelines R746 venaient de contrôles globaux : GitLeaks, Semgrep, Trivy et `pc4_ci_fallback`. Les logs montrent que le fallback ne fait qu’imprimer « Local security scans completed »; il ne compile ni n’exécute Rust.

R747 ajoute un job `r746_cpu_contract` dans le stage `test` par défaut. Il exécute sur le runner PC4, avec l’image Rust 1.90.0 : le harnais CPU R746 (`cargo test --locked`), Clippy et rustfmt. Le travail est ciblé et ne requiert ni GPU, ni modèle, ni service de production. Les trois scanners globaux restent présents dans la première pipeline R747.

Le premier job Rust n’a pas atteint ses commandes : le runner n’a pas résolu `production.cloudfront.docker.com` pendant le pull depuis Docker Hub. Le job a échoué avant le script; ce n’était pas un échec du code Rust. Le job a été corrigé pour tirer la même image Docker officielle depuis Amazon ECR Public (`public.ecr.aws/docker/library/rust:1.90.0-bookworm`). `docker manifest inspect`, avec un répertoire Docker vide, a vérifié le manifeste public du tag. La pipeline de correction #2873, sur le commit `2bc629d65ef055211b4b6381a11f4e3b1e00d62e`, a ensuite tiré cette image sur le runner PC4 et terminé avec succès. `cargo test` : 22 réussis, 0 échec, 3 ignorés parce que le tokenizer externe épinglé n’est pas disponible dans le runner. Clippy et rustfmt ont réussi; Clippy signale deux helpers de compteurs de test SHA-256 déjà connus comme inutilisés. Cela valide le harnais CPU, pas une génération complète, le réseau de production, le GPU ou le modèle entier.

## Contrôles globaux observés

Dans la pipeline initiale #2870, GitLeaks, Semgrep et Trivy sont verts, mais le job CPU a échoué avant son script à cause du pull Docker Hub. Dans la pipeline corrigée #2873, les trois scans et le job CPU sont verts. GitLeaks n’a signalé aucune fuite dans les 887 commits scannés (~160 MB). Semgrep a parcouru 6 898 fichiers avec 303 règles et rapporté 155 findings; son job vert signifie que l’analyse s’est exécutée, pas que les findings sont absents ou triés. Trivy est vert.

## SonarQube et GitGuardian

Aucun connecteur MCP SonarQube ou GitGuardian n’est activé dans cette session. Le projet expose une variable `SONAR_TOKEN` masquée; sa valeur n’a pas été lue ni modifiée, et la liste des variables projet ne contient pas `SONAR_HOST_URL`. Depuis PC6, le port 9000 sur loopback et le nom PC4 ont refusé la connexion. L’endpoint du service n’est donc pas établi ici. GitLeaks est un scanner de secrets utile, mais ne vaut pas une exécution GitGuardian. Aucun secret n’a été recherché dans Gmail.

## Limites et suite

R747 met en place une porte CPU utile dans GitLab. Le tokenizer nécessaire aux trois tests de référence n’est pas conservé dans le dépôt et n’a pas été téléchargé; leur statut demeure ignoré. Il faudra obtenir/configurer un endpoint SonarQube valide avant de l’ajouter à CI et un accès GitGuardian réel avant de revendiquer ce contrôle. La génération complète et l’intégration de ce réducteur à `Session::Reply` restent à construire.
