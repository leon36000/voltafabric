# R636 — borner les dimensions du hook Strata avant tout calcul

## Adaptation

Le hook candidat contre Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` vérifiait `n_tok * k` avant d’avoir borné `n_tok`. R636 remplace cette multiplication de contrôle par `n_tok < 1` puis `n_tok > kMaxWindowEntries / k`, après validation de `k` dans `[1,64]`. La division est donc définie et l’acceptation ne dépend plus d’un produit signé potentiellement débordant. Les boucles et le compteur d’experts ne voient que des dimensions positives dont le produit est au plus 128.

Le patch généré demeure une adaptation opt-in de Strata, sous MIT. Il ne copie pas une nouvelle portion d’un moteur donneur. La réutilisation de GGML/llama.cpp (R626) et de la tête Strata (R627), puis la génération CPU SC117 locale R630, restent les tranches donneuses déjà publiées; R636 durcit le callback CPU candidat de R390–R391.

## Preuve CPU

PC1 possédait déjà Strata au commit exact et ses cinq fichiers donneurs concernés étaient propres. Le générateur a régénéré le patch depuis ce checkout; le vérificateur a confirmé le patch déterministe, son application à un overlay privé et la compilation syntaxique de quatre unités donneuses. Un binaire CPU a ensuite exécuté la branche opt-in extraite avec `n_tok` égal à 0, -1 et `INT64_MAX`. Chaque dispatch a échoué, le callback n’a pas été appelé et le tableau de sortie est resté inchangé. Le test est borné par `MemoryMax=768M`, swap nul, 16 tâches, un cœur et 120 secondes. Aucun worker, modèle, registre R53, réseau de production, génération ou GPU n’a été utilisé; le checkout donneur est resté inchangé.

La première sélection du mode de test a essayé de compiler le donneur complet et a échoué avant exécution car le stub réservé à ce mode n’avait pas été copié. Le mode `shape` a été corrigé pour compiler uniquement la branche extraite, puis le test ciblé a réussi. L’échec initial est conservé dans `receipt.json`.

## Limites

Cette preuve n’établit pas que le callback est installé par le vrai lanceur Strata, ne lie pas le runtime complet, et ne qualifie ni cycle complet de couche/modèle, ni workers mixtes, ni GPU/drain CUDA, ni distribution inter-machine. Le raccord de cycle de vie `PoolMultiFn` reste la prochaine frontière à traiter. Les états `UNKNOWN` précédents ne sont pas rejoués.
