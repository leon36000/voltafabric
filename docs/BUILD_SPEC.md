# VoltaFabric — socle CPU/Rust qualifiable

## Mandat
Construire une extension mince autour de Skinny pour préparer l'inférence unifiée
sur 17 V100 16 Go PCIe, puis inter-PC. Sans qualification GPU implicite. Aucun nouveau
moteur généraliste, forge, cluster, driver, firmware ou service de production.

## Tranche livrée
1. Bibliothèque Rust : contrats MoE SwiGLU pondérés en sortie ; registry versionné
   d'experts/répliques ; dispatch compact global/local ; calcul CPU séquentiel et
   parallèle ; collecteur refusant doublons, erreurs, résultats périmés ou manquants.
2. Budgets mémoire vérifiés et layout NVFP4 block16 explicite (codes + échelles +
   alignement + réserves + pic de chargement), sans implémenter un quantizer GPU.
3. Modèle de liaisons orientées ; comptage de trafic exact sous formats déclarés ;
   placement mémoire/communication heuristique ; partitionnement pipeline contigu.
4. Petit ordonnanceur analytique de DAG à ressources exclusives : concurrence,
   contention et dépendances, clairement étiqueté synthétique, sans tokens/s V100.
5. CLI JSON ; fixtures ; tests différentiels Python/Rust ; procédure de collecte matérielle future
   en lecture seule ; runner LiteLLM de propositions/revues bornées et reprises durables.

## Contrats non négociables
Indices top-k et coefficients inchangés. Aucun remplacement d'expert ni renormalisation.
Chaque slot attendu est accepté une fois ; cela ne promet pas une exécution réseau exactly-once.
Aucune sortie partielle présentée comme une sortie valide. ID de modèle, plan, couche,
lot et epoch dans les résultats. FP64 CPU uniquement dans ce noyau ; référence Python
FP32/FP64 historique séparée. Entrées non finies, dépassements, dimensions et IDs invalides refusés.
GPU/NCCL/RDMA réels : non implémentés ici ; pas de fallback CPU masquant un backend demandé.

## Langages
Rust pour le noyau et CLI ; serde/serde_json pour le protocole (pas de parser maison).
Python standard library pour l'oracle indépendant, les fichiers, HTTP LiteLLM et les preuves.
L'oracle NumPy historique est conservé en référence et rejoué séparément si disponible.

## Sécurité et coûts
Tout dans une racine de projet, aucune écriture système. Travail CPU distant dans un
répertoire de qualification distinct sous Naos ; pas de modification de Skinny existant.
Pas d'appel externe LLM pendant la construction de ce pack. En exploitation : découverte
des aliases réels, clé virtuelle existante, budget de passerelle vérifié, plafond requêtes
et sorties, aucune boucle infinie ni réinjection automatique après timeout ambigu.
Les réponses LLM sont des propositions non fiables, jamais des preuves de tests.

## Acceptation
Rust fmt/check/clippy/tests/debug/release ; tests Python ; oracle croisé et cas invalides ;
exemples CLI réels ; corruption de manifeste détectée ; journaux et sommes SHA256.
Les validations manquantes sont BLOCKED/NOT_RUN, jamais PASS ou SKIP équivalent à succès.
