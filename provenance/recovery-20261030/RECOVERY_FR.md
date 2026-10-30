# Reprise VoltaFabric après reformatage

Le dépôt privé GitLab PC4 est la destination de cette sauvegarde. La branche r364-source-snapshot-20261030 contient le socle R59 hérité de R119, les sources de moteur et de recherche sélectionnées depuis le dépôt de travail PC6, un instantané du registre R53 et les documents de transmission PC6.

La copie du registre R53 est figée : elle ne constitue ni une réservation partagée ni une lease. Avant une utilisation matérielle, restaurer et rapprocher les registres et reçus disponibles sur tous les hôtes ; ne jamais rejouer les identités consommées ou UNKNOWN et réappliquer les admissions matérielles.

Comparaison de 121 fichiers source sélectionnés de l'ancien arbre PC1 avec le dépôt de travail PC6 : 69 identiques, 52 différents, aucun chemin absent sur PC6. Les versions PC1 sont historiques ; cette comparaison ne démontre pas que toutes les données ou tous les autres répertoires de PC1 sont sauvegardés.

Les documents handoff conservent leurs chemins historiques PC1/PC6, qui seront invalides après reformatage. Les modèles, environnements, archives volumineuses, poids, caches et la plupart des reçus ne sont pas inclus. Les tests matériels et une génération complète ne sont pas qualifiés par cette sauvegarde. Pour reprendre, cloner cette branche sur une machine préparée, reconstituer les dépendances et fichiers nécessaires, et vérifier les contrats et admissions. La branche main n'a pas été modifiée.
