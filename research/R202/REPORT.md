# R202 — worker CPU paramétré par les empreintes des résidents

Le worker de recherche R198 est adapté pour recevoir un manifeste explicite d'empreintes de résidents et un répertoire privé de travail, au lieu d'être lié aux seuls artefacts synthétiques R190. Le vrai `PersistentQPN2Owner` vérifie encore l'empreinte et l'identité du résident à chaque chargement. Il accepte les lots de plusieurs tokens et les routes permutées ; le fournisseur et l'alignement restent CPU de test.

Le test emploie le manifeste R190 existant pour deux propriétaires, deux tokens et huit routes permutées, sur deux batches. Les sorties correspondent à la référence R158 et au token nul ; chaque propriétaire ferme après deux EXEC. Commande `/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/source/.work/a92-cpu-venv/bin/python -B research/R202/test_worker.py` : 1 test, exit 0. Run `r202-1b273cd724d347b1826da65b07cfde67`, SHA256 du résultat `a691476dd7e7e67c9141103885b35b19493fc826be50c798ab6a08ef36e5eede`.

Le manifeste doit être fourni par une source de confiance et épinglé par le plan appelant ; cette adaptation ne prouve pas elle-même la provenance. Les huit poids testés sont encore des copies synthétiques d'un expert ; les vrais experts attendent R188 et le montage R192/R200. Pas de GPU, génération, réseau de production ni performance.
