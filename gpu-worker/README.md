# Worker GPU résident MXFP4 — composant de laboratoire R14

Exécutable natif facultatif du protocole `Session` existant. Le cœur Rust conserve
son interdiction d'unsafe et son worker CPU par défaut. Ce projet séparé porte
l'ABI CUDA R13, sans réimplémenter ses kernels ni installer de transport.

Depuis `source`, avec Rust 1.90.0 et une bibliothèque R13 qualifiée pour l'hôte :

```sh
VF_MXFP4_NATIVE_DIR=/dossier/qualifie cargo +1.90.0 build --offline --locked --manifest-path gpu-worker/Cargo.toml --release
CUDA_VISIBLE_DEVICES=GPU-uuid-reserve LD_LIBRARY_PATH=/dossier/qualifie /chemin/vf-mxfp4-gpu-worker --audit-counters
```

Le premier dossier doit contenir `libvf_r13.so`. Il s'agit de la bibliothèque
SM70/CUDA12.8 sur x86-64 V100, ou SM121/CUDA13 sur ARM64 GB10. Les deux artefacts
sont distincts. Les crates ne sont pas vendorisées ; une première résolution
nécessite un cache autorisé ou le réseau. Aucun pilote n'est modifié.

Le binaire vérifie avant HELLO le hash embarqué de la bibliothèque effectivement
mappée, son inode et ses métadonnées. Le fichier et son dossier restent des
artefacts de confiance immuables pendant le processus. Ce contrôle n'est pas un
sandbox pour charger du code natif hostile. `$ORIGIN` permet de placer la
bibliothèque à côté de l'exécutable dans le paquet relocatable.

Le client choisit explicitement
`GpuR13ResidentFp32V1 { library_sha256 }` via `Session::connect_mxfp4`, puis
`load_mxfp4_as`. Le hash attendu doit venir du manifeste qualifié. HELLO et chaque
résultat portent cette identité ; une différence est refusée avant lecture du
corps. Un drapeau GPU passé au worker CPU provoque une erreur d'usage.

Un LOAD crée les propriétaires dans une zone privée, puis publie le binding après
succès de tous les chargements. Un refus d'admission connu ferme le préfixe créé.
Le worker refuse avant `Started` et tout appel de calcul les doublons logiques
`(token, expert)`, y compris dans des slots distincts du protocole brut. Le même
expert reste autorisé pour des tokens différents ; le choix physique doit
correspondre au binding chargé.

Les batches suivants et le rebind d'epoch n'envoient pas les poids. EVICT et
SHUTDOWN libèrent les propriétaires avant leur accusé de réception. Le client
typé conserve sa quarantaine stricte après un LOAD refusé ; le test du rollback
exerce séparément le flux natif, sans ajouter de reprise automatique au client.

Entrées finies exactement BF16 dans des conteneurs FP32 ; coefficient FP32
original appliqué une seule fois avant W2 selon le contrat R9. Les six plans de
poids restent comprimés. Les handles CUDA demeurent sur le thread de service,
avec le contexte exact capturé par R13. Aucun repli CPU. Une erreur CUDA rend le
worker terminal ; aucune sortie partielle ni répétition de la requête.

Les limites actuelles sont celles d'un composant qualifié : un artefact d'une
couche, quatre experts maximum par worker, dimensions multiples de 32 dans
32..5120, 48 millions de MACs par expert et budget cumulé de 100 millions par
EXEC/rang hérité du chemin de référence. Ce dernier budget limite les slots de
grande forme ; son extension exige une qualification séparée. Ces bornes ne
fixent pas le nombre de cartes du futur service.

Qualification R14 : suite CPU complète, exécutions natives debug/release sur V100
et GB10, corpus minuscule indépendant, neuf cas d'un expert réel 5120×2304,
poids chargés une fois, rollback, identités, pannes après démarrage, troncature et
memcheck. Les tests matériels sont ignorés par défaut et exigent une carte
réservée ainsi que `VF_RUN_GPU_TESTS=1`. Ils acceptent `VF_FIXTURE_DIR` et
`VF_GPU_WORKER` pour leur relocalisation. Le test de troncature accepte en plus
`VF_TRUNCATION_PROXY`, un outil privé de laboratoire Python, sans rôle dans le
worker de production.

Les reçus de référence sont dans
`../../.runs/persistent-gpu-worker-20261024-r14/`. Les tests GPU R14 emploient les
pipes locaux de chaque hôte. La connexion SSH utilisée pour piloter la VM n'est
pas une preuve de transport de données GPU inter-PC. Modèle complet, couche
DeepSeek complète, SM120 RTX5080/5090, RDMA, performance distribuée, énergie et
longues sessions restent à qualifier. Le paquet H1 précédent reste gelé.
