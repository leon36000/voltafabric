# R158 — résident Rust et propriétaire persistant sur CPU

À partir de l'échantillon R132 vérifié, le vrai CLI Rust a exporté deux artefacts `QPN2_RESIDENT_V1` distincts avec identités de rank 2 et 9. Chacun contient une copie de test du même expert et pèse 1 770 568 octets ; SHA, identités et reçus sont conservés. Les artefacts sont sous `.runs/r158-real-resident-owner-cpu` sur PC6 et dans un répertoire privé neuf sur PC1, sans être intégrés au dépôt.

Sur PC1, le vrai `PersistentQPN2Owner.load` a vérifié et chargé chaque artefact dans son résident CPU. Un fournisseur de test décode les véritables buffers QPN2 reçus et réalise les produits CPU en binaire64 avec étapes FP16 ; aucun kernel CUDA n'est utilisé. Chaque owner a effectué deux EXEC de numéros 1 et 2 sur les mêmes buffers résidents : quatre calculs acceptés, quatre projections synchronisées par owner, sorties FP16 identiques et pointeurs des poids stables. Une répétition du numéro d'EXEC est refusée.

Une erreur CPU synthétique injectée après admission sur le premier owner provoque `POISONED_UNKNOWN`, le refus d'une nouvelle exécution et la fermeture `POISONED_UNKNOWN_CLOSED`. Le second owner se ferme normalement. Le résultat inconnu synthétique n'a pas été rejoué. Les deux sorties nominales concordent bit à bit avec la ligne qualifiée R156, puis le vrai collecteur Rust vérifie les 2 048 valeurs et applique les coefficients une seule fois. Les étapes d'export, routage, exécution et vérification sortent 0.

Ce lot qualifie le chargement d'artefact et le cycle de vie **CPU** du propriétaire Python avec fournisseur simulé. Il ne qualifie pas Rust Session/Collector en processus de production, réseau, CUDA, modèle complet ou performance ; les deux experts sont des copies synthétiques d'un seul échantillon et le SHA complet du shard reste inconnu.

Suite : relier ces owners au protocole/session réel et aux reçus de durée de vie, ou qualifier séparément les dépendances de 1Cat avant une éventuelle admission GPU.
