# Vérifier l'instantané GitLab R59

Depuis la racine réelle du dépôt :

```sh
cd -P /chemin/vers/le/depot
python3 -B scripts/check_gitlab_snapshot.py
```

Le contrôle lit `provenance/source_final_r59.json`, exige son SHA256 épinglé
`41f179ce7d7076748e907e566182aa1d3298b4deb776cb52ecd0f694edba526d`, puis
vérifie les 370 fichiers qui y sont listés. Il refuse les chemins non canoniques,
les liens symboliques dans le chemin, les fichiers spéciaux, absents ou altérés,
et borne chaque lecture. Le répertoire du dépôt et ses parents doivent eux-mêmes
être accessibles sans traverser de lien symbolique ; le raccourci du Bureau est
un lien : utiliser le vrai chemin du dépôt, par exemple avec `cd -P`.

Ce script exige les interfaces POSIX `dir_fd`, `O_DIRECTORY` et `O_NOFOLLOW` ;
il est indépendant du chemin d'installation, mais n'est pas qualifié sur Windows.
Il vérifie la stabilité de chaque fichier pendant sa lecture et ses composants de
chemin, sans construire un instantané atomique global des 370 fichiers : éviter
un écrivain concurrent. Il ne constitue pas une isolation contre un adversaire
disposant du même compte.

Un résultat positif ne concerne que les fichiers énumérés par ce manifeste. Les
fichiers supplémentaires, le registre partagé vivant, les archives complètes,
les modèles, les portes CPU ou les calculs GPU ne sont pas qualifiés. Il ne
remplace ni ne modifie le `SHA256SUMS` historique ni `scripts/check_integrity.py`.
