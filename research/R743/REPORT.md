# R743 — Scénarios concurrents dans le planner CPU

## Changement

Le planner évaluait chaque demande isolément. R743 ajoute une API facultative pour fournir des groupes explicites de demandes simultanées. Les DAG des demandes sont fusionnés sous des identifiants déterministes; les ressources GPU de rang et de lien restent partagées entre les graphes. Le scheduler peut donc recouvrir le calcul sur des rangs distincts et sérialiser le travail qui réclame la même ressource exclusive.

`ConcurrentDemandScenario` référence les demandes existantes par indices triés pour l'évaluation. Les routes et leurs choix d'experts ne sont ni réécrits ni renormalisés. Les poids restent résidents: chaque candidat passe par `Prepared::memory`, avec refus si la capacité ne suffit pas. Les fréquences de scénario s'ajoutent comme poids à la moyenne exacte des demandes isolées.

Les scénarios sont bornés à 256 groupes, de 2 à 16 demandes chacun, avec indices uniques valides et fréquence positive finie. Chaque graphe combiné est limité à 4096 jobs et consomme le `WorkBudget` existant. Une entrée invalide, un manque de capacité ou l'épuisement du budget renvoie une erreur sans plan partiel. Les appels `plan` existants conservent leur forme JSON: le nouveau champ de diagnostics est omis quand aucun scénario n'est fourni.

## Source de l'idée

HeteGen décrit une exécution CPU/GPU hétérogène avec chevauchement asynchrone pour réduire les goulots d'I/O ([arXiv:2403.01164](https://arxiv.org/abs/2403.01164)). R743 adapte seulement l'idée d'évaluer explicitement des DAG qui se chevauchent. Aucun code HeteGen n'est repris; VoltaFabric ne reproduit pas son offload CPU.

Le coût reste le makespan du scheduler déterministe à ressources exclusives de VoltaFabric. Il ne simule ni partage fluide de bande passante, ni files d'attente réelles, ni latences mesurées. Il ne qualifie donc aucun GPU ni gain de vitesse.

## Vérification CPU et publication

- Harness Rust isolé avec planner et modules de contrat R742: 5 tests réussis, 0 échoué.
- Cas couverts: chevauchement sur GPU indépendants, sérialisation sur GPU partagé, placement de poids résidents, refus de capacité, scénarios mal formés, forme JSON historique et épuisement du budget sans plan partiel.
- `rustfmt --edition 2021 --check` passe sur le planner et le nouveau test.
- Le harness local utilise `serde_json 1.0.154`, présent dans le cache hors ligne; le verrou de dépendances du dépôt n'a pas été modifié.
- Lecture en retour exacte des sept fichiers R743 au commit corrigé `b9e7776261acb88b91e657b366784f088eb25788`.
- Pipeline source #2621 de ce commit réussi: Gitleaks, Semgrep, Trivy et fallback CI.
- Le premier commit d'artefacts `3aefc412576f7ff46b1e131a3457533f72541a41` a encodé par erreur les fichiers comme texte Base64; son pipeline #2620 a été annulé. Le commit descendant `b9e7776261acb88b91e657b366784f088eb25788` a restauré le texte source. L'historique n'a pas été réécrit.

## Portée

Pas de génération du modèle, de chargement de checkpoint, de calcul GPU, de transport inter-PC, de réservation R53 ou de mesure de performance. Il s'agit d'une adaptation analytique d'ordonnancement, sans code logiciel donneur.

## Suite

Poursuivre une frontière CPU bornée entre le checkpoint indexé, les propriétaires résidents et le Session existant, avec refus atomique de données de tenseur incomplètes. Ce planner analytique ne remplace ni l'oracle numérique ni les admissions matérielles.
