# R614 — contrôle en lecture seule du registre R53

La sauvegarde figée `provenance/recovery-20261030/namespace-ledger` contient 80 entrées, dont 78 `RESERVED_UNKNOWN_NO_REPLAY` et deux `CONSUMED_RETROSPECTIVE_CPU_NO_REPLAY`. Son manifeste ordonné nom/NUL/SHA-256 vaut `8a3dce74f5d9b815437c7f0c12372dcf0b355014d1dbad94877a331681482e40` pour les octets présents. Ce hash permet une comparaison avec un exemplaire de référence préalablement choisi ; il n'authentifie pas lui-même la provenance ni la complétude historique de la sauvegarde.

`reconcile.py` lit le snapshot et un registre candidat sans les modifier. Il exige le hash externe attendu du snapshot, refuse les liens symboliques et physiques, les entrées non régulières ou hors limite et les permissions insuffisamment privées du candidat. Il contrôle l'identité des entrées du snapshot, détecte les absences et les octets modifiés, signale les entrées supplémentaires et retourne l'identité device/inode/uid observée du répertoire candidat. Un résultat positif signifie seulement que les octets des 80 entrées sauvegardées sont couverts à cet instant ; il ne valide jamais la lease, le caractère partagé, l'exhaustivité des reçus des autres hôtes ou l'admission matérielle. Les entrées supplémentaires restent à réconcilier, y compris les UNKNOWN.

Dix tests passent, notamment fichier manquant, modification, hash de snapshot incorrect, entrées et répertoire symlink, hardlink, permissions et clés JSON dupliquées. Sur une copie locale de test des 80 enregistrements, la couverture est positive ; une seconde copie avec une entrée manquante et une modifiée est refusée. Ces copies sont des fixtures de validation et ne sont ni un registre actif ni une autorisation de lancement. Le chemin historique PC6 indiqué dans le handoff n'existe pas actuellement ; une recherche bornée des répertoires `namespace-ledger` sous `/home/pc1/NAOS`, profondeur quatre, n'en a pas trouvé. Cela ne prouve pas l'absence d'un registre ou d'une sauvegarde ailleurs.

Utilisation indicative, avec un hash obtenu d'une source de confiance indépendante :

```
python3 research/R614/reconcile.py SNAPSHOT CANDIDAT --expected-snapshot-sha256 HASH
```

Avant tout worker inter-PC ou GPU, il reste à identifier le véritable registre partagé et sa lease coopérative, rapprocher les reçus de tous les hôtes, puis satisfaire les autres admissions. Ce contrôle ne crée ni ne restaure de registre et ne rejoue aucune opération.
