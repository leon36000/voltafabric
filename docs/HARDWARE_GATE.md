# Qualification matérielle — portes restantes

État au 20 octobre 2026 : A81 a mesuré la V100 PC3 sur PCIe Gen3 x4 et son profil
de transferts; A95 a passé un canary GPU sur un expert Qwen avec le paquet H1
relocatable. Ces preuves ne ferment ni la couche DeepSeek complète, ni H2/H3.
La preuve physique A81 vient du `lspci` hôte; le lien négocié était 8 GT/s x4,
capacité 8 GT/s x16, au repos et sous charge. Ses reçus et limites sont dans
`evidence/continuation_a81_pcie_x4/PHASE_REPORT.md`. Avant chaque nouvel essai,
reconfirmer les allocations et l'idle du GPU exact; ne pas déduire qu'une mesure
historique réserve encore la carte. Ne pas monopoliser les 17 GPU; commencer par
une carte canary, puis deux, quatre et davantage selon preuves.

## Relever sans modifier

Inventaire réel UUID/BDF/SM/mémoire, topologie `nvidia-smi topo -m`, `lspci -t/-vv`,
versions pilote/CUDA/NCCL/NIC, largeur PCIe négociée sous charge, CPU/NUMA et santé.
Lire les paramètres ne justifie pas de les modifier. Une étiquette PCIe x16 ou RDMA
ne prouve pas la bande passante ni le trajet zéro-copy.

## Compilation SM70 reproductible

`cuda/build_sm70.sh` est reproductible au bit depuis A23 : cinq compilations
indépendantes, dans des répertoires différents dont un avec espaces, donnent la même
bibliothèque `85a49f50…` (115784 octets). Avant A23, nvcc incorporait le nom de son
intermédiaire `tmpxft_<pid>_…` et chaque build différait de ces octets seuls; le hash
livré n'était donc pas re-dérivable par un destinataire, et `package_h1.py` comme
`h1_preflight.py` refusaient toute recompilation légitime. Le code machine est resté
identique : la section `.nv_fatbin` des bibliothèques pré-A16, A16 et A23 est la même.
Un hash différent du reçu signifie donc maintenant une vraie divergence, pas un aléa
de compilation. `VF_KEEP_DIR` peut être déplacé sans changer les octets produits.

## Exécuter dans cet ordre

1. Compilation SM70 de l'adaptateur minimal; dépendances épinglées compatibles.
2. Oracle numérique kernel seul puis couche MoE, différentes formes et precisions.
3. Deux GPU : index global/local, experts vides, déséquilibre, retards, saturation,
   invalidation epoch, graphes capturés, buffers et reprise contrôlée.
4. Mesure P2P dans les deux directions, lectures/écritures si pertinentes, petites et
   grandes tailles, flux simultanés et ressources PCIe communes.
5. Mesure GPU↔NIC et inter-PC, contrôle du chemin réellement utilisé, repli explicite
   host-staged si nécessaire. ConnectX-3 ne devient pas ConnectX-4 par configuration.
6. TP+PP corrigé versus EP/hybride, mêmes poids et protocole. Le MTP/pipeline exige les
   bons tokens et identités à chaque stage; tester acceptation, rejet et graph replay.
7. Service réel longue session et pannes contrôlées; coût énergétique total.

## Interdictions sans nouveau feu vert

Pas de flash BIOS/VBIOS, modification ACS/IOMMU/firmware, augmentation de puissance,
changement de pilote/noyau, reboot ou achat NTB. Pas de changement GPU de production,
pas de nettoyage d'autres projets, pas de reconstruction PC5/RAG/corpus/Quanta.

## Résultats attendus

Rapports séparés capacité / correction / communication / qualité / énergie. Provenance
par UUID, date et hash; aucune mesure synthétique promue en mesure réelle. L'installation
ne devient production qu'après validation du service complet, pas après `cargo test`.

## Commande du canary H1 complet

Le canary A95 a vérifié un expert Qwen sur la V100 PC3; le critère H1 reste ouvert
pour la couche complète DeepSeek, les formes restantes, la mémoire et le service.
Pour un nouveau canary correspondant à ce contrat, après inventaire, disponibilité
et réservation d'une seule V100, utiliser une pile Python GPU existante : Torch
CUDA12 déclarant sm_70/compute_70, pas le venv CPU.
Le launcher exige VF_GPU_PYTHON (chemin réel) et VF_GPU_UUID (UUID réservé).
VF_GPU_DEVICE vaut 0 par défaut, VF_H1_RUN_ID peut donner un nom de reçu inédit.

```sh
VF_GPU_PYTHON=/chemin/vers/python-gpu-verifie VF_GPU_UUID=GPU-uuid-v100-verifie bash handoff/hardware_canary.sh
```

Les deux valeurs ci-dessus sont des paramètres à résoudre au matériel, jamais des
identités déjà vérifiées. Depuis A24 le launcher part de la bibliothèque qualifiée
`build/libvf_qpn2_moe_sm70.so` et passe `--cuda-proof`, par défaut
`evidence/continuation_a23/CUDA_COMPILED.json` (`VF_H1_LIBRARY` et `VF_H1_CUDA_PROOF`
restent surchargeables). Avant A24 le défaut pointait sur l'artefact A123
`3cfaa80e…`, qualifié par un reçu dépassé, et rien ne le vérifiait : un premier run
V100 aurait inscrit dans le reçu matériel un hash correspondant ni au paquet ni à la
provenance courante. `hardware_canary.py` refuse maintenant, avant tout appel CUDA,
une bibliothèque qu'aucun reçu CUDA_COMPILED sm_70 non exécuté ne qualifie; le reçu
H1 enregistre le chemin et le hash du reçu de compilation utilisé. Dans le paquet,
`run_h1.sh` impose la bibliothèque et le reçu du bundle. Le script charge la FFI compilée, garde les poids compressés,
vérifie UUID/SM70/CUDA12, emploie le stream du device et synchronise avant comparaison.
Six configurations GEMM dyadiques exigent égalité exacte ; deux mappings avec
répliques, deux modèles logiques construits par Rust (trois rangs dont un vide),
et trois crops réels ont des seuils déclarés avant exécution dans le code.
Depuis A151, deux cas supplémentaires raccordent `moe_apply_local` à
`resident-completions` puis au collecteur Rust, avec sorties brutes non pondérées,
identité exacte des slots et synchronisation avant publication. Les quinze cas
préparés prévoient 29 appels kernel (hors options full-expert) ; aucun de ces appels
n’est une preuve matérielle avant exécution. Le raccord A151 est une validation de
frontière, pas un worker GPU persistant : ce canary reconstruit et revalide encore
les poids à chaque appel. Le paquet A150 gelé conserve ses treize cas/21 appels. Les modèles Rust gardent leurs poids
compressés et passent leurs sorties brutes au collecteur strict.
Tout dépassement rejette le candidat ; ne pas élargir ces seuils pour le faire passer.

Le reçu conserve versions, hashes, cas et mémoire ; GPU_CANARY_PASS ne ferme pas H1
(capture, autres formes, couche complète et mémoire restent à qualifier).
Un reçu matériel existant est refusé avant tout accès CUDA : l'inspecter ainsi que
la session/operation Naos avant de choisir un nouveau run. Aucun replay aveugle.
