# R747 — exécuter le contrat CPU dans le runner GitLab PC4

## Changement CI

Le projet privé n’avait pas de `.gitlab-ci.yml` à la racine. Les pipelines R746 venaient de contrôles globaux : GitLeaks, Semgrep, Trivy et `pc4_ci_fallback`. Les logs montrent que le fallback ne fait qu’imprimer « Local security scans completed »; il ne compile ni n’exécute Rust.

R747 ajoute un job `r746_cpu_contract` dans le stage `test` par défaut. Il exécute sur le runner PC4, avec l’image Rust 1.90.0 : le harnais CPU R746 (`cargo test --locked`), Clippy et rustfmt. Le travail est ciblé et ne requiert ni GPU, ni modèle, ni service de production. Les trois scanners globaux restent présents dans la première pipeline R747.

Le premier job Rust n’a pas atteint ses commandes : le runner n’a pas résolu `production.cloudfront.docker.com` pendant le pull depuis Docker Hub. Le job a échoué avant le script; ce n’est pas un échec du code Rust. Le job a été corrigé pour tirer la même image Docker officielle depuis Amazon ECR Public (`public.ecr.aws/docker/library/rust:1.90.0-bookworm`). `docker manifest inspect`, avec un répertoire Docker vide, a vérifié le manifeste public du tag. La relance doit encore prouver le pull depuis le runner et l’exécution réelle des tests.

## Contrôles globaux observés

Dans la pipeline R747 #2870, GitLeaks, Semgrep et Trivy sont tous verts. GitLeaks n’a signalé aucune fuite. Semgrep a parcouru 6 898 fichiers avec 303 règles et rapporté 155 findings; son job est vert malgré ces findings, qui ne sont pas triés ici. Trivy est vert. Le statut global de cette pipeline est `failed` uniquement parce que l’image du job CPU n’a pas pu être tirée.

## SonarQube et GitGuardian

Aucun connecteur MCP SonarQube ou GitGuardian n’est activé dans cette session et l’usage du navigateur local est désactivé. Le projet expose une variable `SONAR_TOKEN` masquée; sa valeur n’a pas été lue ni modifiée, et la liste des variables projet ne contient pas `SONAR_HOST_URL`. Depuis PC6, le port 9000 sur loopback et le nom PC4 ont refusé la connexion. L’endpoint du service n’est donc pas établi ici. GitLeaks est un scanner de secrets utile, mais ne vaut pas une exécution GitGuardian.

## Suite

Lire la nouvelle pipeline après le commit de correction. Confirmer que `r746_cpu_contract` tire l’image ECR, exécute les trois commandes et que les scans globaux restent actifs. Ne pas promouvoir la porte avant ce résultat.
