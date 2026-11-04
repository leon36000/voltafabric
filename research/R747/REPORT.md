# R747 — exécuter le contrat CPU dans le runner GitLab PC4

## Changement

Le projet privé n’avait pas de `.gitlab-ci.yml` à la racine. Les pipelines R746 provenaient de contrôles globaux : GitLeaks, Semgrep, Trivy et `pc4_ci_fallback`. Les logs montrent que ce dernier ne fait qu’imprimer « Local security scans completed »; il ne compile ni n’exécute Rust.

R747 ajoute une configuration GitLab minimale avec le stage `test` par défaut et un job `r746_cpu_contract`. Sur le runner PC4 en image Rust 1.90, il exécute le harnais CPU R746 avec le lockfile, Clippy et rustfmt. Le job est séparé de la compilation de l’application complète et n’a pas besoin de GPU, de modèle ou de service de production. La première pipeline de R747 vérifiera si les analyses globales imposées par le serveur restent actives avec la configuration de dépôt.

## État des protections existantes

Sur les pipelines R746 #2702 et #2703, les quatre jobs sont verts. GitLeaks 8.30.1 a parcouru 885 commits (~160 MB) et indiqué « no leaks found ». Trivy 0.76.1 a fini son analyse vulnérabilités/configuration. Semgrep 1.182.0 a parcouru 6 898 fichiers avec 303 règles et rapporté 155 findings; son job est vert malgré ces findings. Ils ne sont pas triés ici et le vert ne signifie pas zéro finding. Le job fallback ne compte pas comme une porte de test.

## SonarQube et GitGuardian

Aucun outil MCP SonarQube ou GitGuardian n’est activé dans cette session et le navigateur local est désactivé. Le projet contient une variable `SONAR_TOKEN` masquée; sa valeur n’a pas été lue ni modifiée, et aucun `SONAR_HOST_URL` de projet n’est défini. Depuis PC6, les tests HTTP du port 9000 sur loopback et le nom PC4 ont refusé la connexion. Cela ne prouve pas l’absence du service ailleurs; son endpoint n’est pas établi ici. GitLeaks est une détection de secrets utile, mais ne vaut pas une exécution de GitGuardian.

## Vérification attendue

Le commit R747 déclenche la pipeline PC4. Contrôler le statut du nouveau job `r746_cpu_contract`, le statut des scans globaux et les logs de test. Aucun résultat de pipeline n’est revendiqué dans ce rapport avant cette exécution.
