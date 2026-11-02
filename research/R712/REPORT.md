# R712 — charge d’un propriétaire DeepSeek dans la Session CPU existante

## Frontière raccordée

R711 construisait un `session::mxfp4::Artifact` compact à partir des six tensors MXFP4 d’un expert choisi. R712 ajoute `PinnedMxfp4RankResidentExpert::load_into_session`: la méthode consomme le reçu résident, reverifie l’identité route/attempt et le SHA du propriétaire compact, puis utilise `Session::load_mxfp4` et conserve le `LoadReceipt` sous l’identité complète run/modèle/plan/couche/batch/epoch/attempt. L’API Session existante reste responsable de l’HELLO, du rang, du SHA et du LOAD.

Le test démarre un vrai processus local `vf-rank-session`, charge l’expert 0 d’une route top‑6 synthétique, vérifie le reçu et ferme le worker. Aucun EXEC n’est envoyé: les cinq autres experts ne sont pas résidents dans ce test. La valeur consommée ne peut pas servir à rejouer automatiquement un LOAD incertain. Un reçu LOAD individuel ne devient jamais une preuve de résultat de couche.

## Vérifications CPU

- `cargo test --offline --locked --test r712_deepseek_session_owner`: 1 réussi.
- `cargo test --offline --locked --release --test r712_deepseek_session_owner`: 1 réussi.
- Tests de lecture/packing `a148_pinned_expert_selective_read_tests`: 5 réussis en debug et 5 en release.
- `cargo clippy --offline --locked --lib -- -D warnings`: réussi.
- `cargo clippy --offline --locked --test r712_deepseek_session_owner -- -D warnings`: réussi.
- `rustfmt --edition 2021 --check src/deepseek_v41.rs tests/r712_deepseek_session_owner.rs`: réussi.
- `git diff --check -- src/deepseek_v41.rs tests/r712_deepseek_session_owner.rs`: réussi.

Le test utilise les métadonnées officielles épinglées et des octets de poids synthétiques dans un lecteur clairsemé. Aucun poids réel du checkpoint n’a été chargé.

## Réemploi et limites

Réemploi interne effectif: Safetensors, `session::mxfp4::Artifact`, `Session::connect_mxfp4`, `Session::load_mxfp4`, `LoadReceipt` et le worker CPU déjà présent. Aucun code d’un moteur donneur n’a été copié dans cette tranche. Le reçu ajouté restitue l’identité que le protocole LOAD historique ne porte pas.

Cette étape qualifie un LOAD CPU d’un seul artefact. Elle ne qualifie pas le préflight atomique de la route complète, une génération, un modèle, les poids réels, l’exécution GPU/V100, le transport PC6–PC4, RDMA, la performance, ni le raccord au checkpoint complet. R53 de production n’a pas été utilisé. Aucun UNKNOWN n’a été rejoué. La relance automatique reste arrêtée par le propriétaire.

## Publication

La publication R712 attend une lecture fraîche de la branche R710 sur GitLab. Le connecteur PC4 renvoie `fetch failed`; la route Tailnet de PC4 est hors ligne. L’adresse directe de PC4 répond au ping, mais son HTTPS présente le certificat Traefik auto-signé par défaut et l’HTTP renvoie 404. Aucun contournement TLS, écrasement de branche ou publication n’a été effectué. Le dernier commit de base connu est `f4c73db055f63b5db014c334d082ae414efb2541`; il devra être revérifié avant la création d’une branche successorale.

## Suite

R713: étendre le préflight local au jeu top‑6 et à ses bindings de rang avant tout démarrage de worker; refuser sans effet si un tensor, hash, rang ou budget de résidence manque. Garder les sorties atomiques et ne pas envoyer EXEC avant que chaque routeur expert soit résident. Cette suite reste CPU synthétique tant qu’un manifeste de poids réels et une admission de mémoire explicite ne sont pas disponibles.
