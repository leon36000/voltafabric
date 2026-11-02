# R743 — Scénarios concurrents dans le planner CPU

## Changement livré

Le planner évaluait chaque demande isolément. R743 ajoute une API facultative pour fournir des groupes explicites de demandes simultanées. Les graphes de chaque demande sont fusionnés sous des identifiants déterministes; les GPU de rang et les ressources de lien existantes restent partagés entre les graphes. Le scheduler peut donc recouvrir du calcul sur des rangs distincts et sérialiser les travaux qui réclament la même ressource exclusive.

`ConcurrentDemandScenario` référence les demandes existantes par indices triés pour l'évaluation. Les routes et leurs choix d'experts ne sont ni réécrits ni renormalisés. Les poids restent résidents: chaque évaluation, déplacement et échange passe toujours par `Prepared::memory`, avec refus si la capacité ne suffit pas. Les fréquences des scénarios sont des poids additionnels dans la moyenne exacte; elles s'ajoutent aux fréquences des demandes isolées.

Les scénarios sont bornés à 256 groupes, de 2 à 16 demandes chacun, avec indices uniques valides et fréquence positive finie. Les graphes combinés restent dans la limite de 4096 jobs et consomment le WorkBudget existant. Une entrée invalide, un manque de capacité ou l'épuisement du budget renvoie une erreur sans plan partiel. Les appels `plan` existants conservent leur sortie JSON: le nouveau champ de diagnostics est omis quand aucun scénario n'est demandé.

## Source de l'idée et adaptation

HeteGen décrit le chevauchement asynchrone CPU–GPU comme une technique pour réduire les attentes d'I/O ([arXiv:2403.01164](https://arxiv.org/abs/2403.01164)). Nous en reprenons uniquement l'idée de comparer explicitement des graphes qui se chevauchent. VoltaFabric ne copie pas son code et ne reproduit pas son offload CPU. Ici, le coût reste le makespan du scheduler déterministe à ressources exclusives de VoltaFabric. Il ne simule pas le partage fluide de bande passante, les files d'attente réelles ou un temps mesuré; les résultats sont analytiques et ne qualifient aucun GPU.

## Vérification

- `rustfmt --edition 2021 --check` sur `src/planner.rs` et le nouveau test: succès.
- Harness CPU isolé construit avec le planner R743 et les modules contractuels R742: 5 tests réussis, 0 échoué.
- Les tests couvrent le recouvrement sur deux GPU indépendants, la sérialisation sur un GPU partagé, le déplacement d'experts résidents, les refus de capacité/entrée, la conservation des routes et la sortie sans plan partiel à l'épuisement du budget.
- Le harness local utilise `serde_json 1.0.154`, présent dans le cache hors ligne; la version verrouillée du dépôt reste inchangée. Le pipeline GitLab complet vérifiera l'intégration avec le verrou du projet.
- R742: pipeline source #2610 et pipeline reçu #2612 en succès; R741: pipeline reçu #2609 en succès.

## Portée

Aucune génération de modèle, aucun chargement de checkpoint, aucun calcul GPU, aucun transport inter-PC, aucune réservation R53 ou mesure de performance. Cette tranche ne copie aucun composant logiciel HeteGen; elle adapte une idée de planification à l'ordonnanceur CPU existant.

## Suite

Après la lecture exacte des six fichiers et le succès des contrôles GitLab, reprendre une frontière de génération qui utilise les propriétaires résidents, avec une fixture indépendante et un refus atomique en cas de données manquantes. Ce modèle analytique de planification ne remplace pas une mesure matérielle.
