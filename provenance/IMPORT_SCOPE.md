# Import initial GitLab de VoltaFabric

Ce commit est un instantané initial, et non la reconstruction d'un historique Git antérieur. Les 370 fichiers fonctionnels proviennent du manifeste R59 dans `provenance/source_final_r59.json` : leur contenu est inchangé et les fixtures et licences du manifeste sont incluses. Les documents supplémentaires et les états `handoff/` sont des instantanés séparés, non couverts par le manifeste R59.

`provenance/namespace-ledger-snapshot/` est une copie documentaire des identités déjà consommées à la date de l'import. **Elle ne remplace pas le registre partagé original** et ne permet pas à un clone de réserver ou de rejouer un test. Tous les résultats UNKNOWN restent consommés. Avant tout matériel, employer les gardes, leases et le registre partagé effectifs, avec une nouvelle identité.

Le dépôt ne contient ni les cibles de compilation, environnements locaux, modèles complets, binaires GPU privés, caches, tous les reçus/archives `.runs` ou la sauvegarde PC4. Les rapports sous `research/` sont des extraits de référence, pas des campagnes complètes. Les chemins absolus dans les états et rapports désignent l'installation source sur PC6/PC1/PC3 et ne prouvent pas qu'un clone sur une autre machine puisse s'exécuter.

Le fichier SHA256SUMS initial et scripts/check_integrity.py concernent un état historique et ne constituent pas un contrôle de l'état R59. Ne pas réécrire cette histoire ; pour contrôler cet instantané, comparer les fichiers listés dans `provenance/source_final_r59.json`. Aucun CPU_FULL_PASS récent, modèle complet, gain de performance ou qualification GPU globale n'est déduit de ce commit.

Les documents `PROMPT_CODEX.md`, `evidence/DELIVERY.json` et `evidence/RESULTATS.md` sont inclus pour résoudre les références des documents du projet ; ils décrivent une livraison historique et ne remplacent pas le manifeste R59 ni les résultats récents.
