# Classement et récupération VoltaFabric

L’index de référence est `handoff/STORAGE_INDEX.json`, lié par son SHA-256 au checkpoint et aux tâches. Le chantier actif reste `/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/source` sur PC6. Les sources, poids, outils, bibliothèques de build, dernières preuves et identités consommées restent accessibles localement.

R66 a supprimé uniquement 60 caches incrémentaux Rust régénérables et trois petits squelettes vides ou réduits à des marqueurs Cargo. Reçus dans la racine du chantier : `/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/storage-retention-r66-20261026/operation-r2.json` et `/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/storage-retention-r66-20261026/placeholder-removal.json`. Aucun dossier historique incertain, source, poids, oracle ou binaire de preuve n’a été supprimé.

PC4 est actuellement inaccessible. Aucune sauvegarde n’y a été créée. Le disque annoncé de 8 To n’a donc ni UUID, ni point de montage, ni destination validés dans l’index. Ne pas interpréter le classement proposé ci-dessous comme une archive existante.

Classement relatif prévu, sous le vrai point de montage qui devra être vérifié :

```text
Projets en cours/
  VoltaFabric/
    00_INDEX_ET_RESTAURATION/
    01_SOURCES_ET_ETAT/
    02_PREUVES_ET_ORACLES/
    03_PAQUETS_H1_ET_WORKERS/
    04_ARCHIVES_HISTORIQUES/
      PC6/
```

Chaque archive devra avoir un identifiant, la liste des chemins d’origine, une révision/manifeste source, un emplacement absolu réel, l’UUID du disque, les tailles, un manifeste SHA-256, un reçu de vérification et un test de restauration. Copier son entrée dans l’index local et dans le catalogue distant avant d’enlever une copie locale. Le moteur actif ne doit jamais dépendre implicitement de la présence du disque d’archive de PC4.

Les historiques dont l’utilité future est incertaine restent sur PC6 tant que cette chaîne de preuve n’est pas satisfaite. Ne jamais supprimer une preuve sur la seule base de son âge ou d’un nom de dossier ancien. Les caches incrémentaux supprimés pourront être régénérés par le compilateur ; leur suppression peut rallonger une compilation ultérieure.
