# VoltaFabric 0.1.0 — socle CPU/Rust et transmission à Codex

**Livraison du 7 octobre 2026. Code compilé et testé sur CPU ; aucun kernel V100 ni
transport PCIe/RDMA n'est qualifié par cette livraison.**

## Démarrer avec Codex

Extraire le dossier `VoltaFabric` dans un emplacement dédié sous la racine Naos déjà
retenue. Ne pas écraser un chantier existant. Ouvrir ce dossier comme workspace Codex,
puis coller le contenu de `PROMPT_CODEX.md` dans la conversation. `AGENTS.md` contient
les règles persistantes, le prompt la mission et les prochaines tranches exécutables.
Il n'est pas nécessaire de recopier l'historique des conversations.

Le prompt demande de reproduire les preuves avant de modifier, puis d'intégrer le
mapping MoE au Skinny réellement présent. Il prévoit des vagues d'agents outillés et/ou
de propositions LiteLLM, sans reconstruction de Forgejo, GitHub ou de l'infrastructure.

## Preuves héritées — qualification initiale

| Vérification réellement exécutée | Résultat | Preuve |
|---|---|---|
| Rust debug | 61 tests, zéro échec/ignoré | `evidence/remote/first_green.log`, `final_gate.json` |
| Rust release | les mêmes 61 tests, zéro échec/ignoré | `evidence/remote/release.log`, `final_gate.json` |
| Rust format + Clippy avec warnings interdits | succès | `evidence/remote/final_gate.json`, `clippy.log` |
| Oracle indépendant Python → binaire Rust | 923 cas, succès | `evidence/remote/crosscheck.json` |
| Runner de propositions, serveur HTTP simulé et reprises | 45 tests, succès | `evidence/current/python_final.log` |
| Réexécution de l'oracle NumPy hérité | 32 tests, succès | `evidence/current/inherited_numpy_rerun.log` |
| Intégrité du code transféré depuis PC1 | SHA-256 identique | `evidence/remote/RUST_SOURCE_SHA256SUMS` |

Les 61 tests Rust ne deviennent pas 122 tests distincts parce qu'ils passent dans deux
profils. Les 923 cas d'oracle ne sont ni 923 modèles ni 923 benchmarks matériels.
L'erreur maximale observée est 1,4210854715202004e-14 dans les cas CPU f64 exécutés,
avec atol=rtol=2e-11. Ce résultat ne qualifie pas la précision d'un vrai LLM quantifié.

`evidence/DELIVERY.json` est le résumé machine ; `evidence/RESULTATS.md` expose la portée.
Les journaux rouges sont conservés dans `evidence/tdd/` et `evidence/remote/red_memory.log`
pour les régressions effectivement reproduites. Toute la suite n'a pas une trace rouge
individuelle : ne pas prétendre le contraire.

## Reproduire

Prérequis : Rust 1.90.0 avec rustfmt et Clippy, Python 3.11+ ; aucun GPU. Les crates sont
verrouillés par Cargo.lock mais pas vendoriés. Une première compilation exige donc un
cache valide ou l'accès au registre. Ne pas confondre « sans V100 » et « sans réseau ».

À la racine du projet, avant toute modification :

```sh
python3 scripts/check_integrity.py
python3 scripts/verify.py
```

Les tests de l'adaptateur placent leurs dossiers temporaires dans `TMPDIR` quand il est
exporté, sinon sous `evidence/adapter-temp` : aucune variable d'environnement
supplémentaire n'est requise et les temporaires ne retombent jamais dans `/tmp`.

`verify.py` produit un dossier de preuves unique dans `evidence/runs/`, exécute les tests
Python, le dry-run, fmt, Clippy, debug, release, build et l'oracle indépendant. Une étape
absente/échouée donne FAIL. `--python-only` permet un diagnostic partiel et ne produit
jamais le statut CPU_FULL_PASS. Le nom du run doit être nouveau à chaque exécution.
Les étapes Rust et Python ont été qualifiées sur deux environnements pendant la
construction ; le script complet est fourni pour les réunir sur l'hôte Codex.

Exemples de commandes sans GPU :

```sh
cargo run --locked -- capabilities
cargo run --locked -- moe fixtures/moe_seed_1.json
cargo run --locked -- layout fixtures/layout.json
cargo run --locked -- traffic fixtures/traffic_grouped_win.json
cargo run --locked -- plan fixtures/placement_SYNTHETIC.json
cargo run --locked -- pipeline fixtures/pipeline.json
```

Le client `session-run-mxfp4` accepte `startup_parallelism` de 1 à 4 dans son entrée
JSON. Le défaut 1 conserve le démarrage séquentiel. Une valeur supérieure fait se
chevaucher uniquement les connexions et LOAD de workers indépendants, par vagues
bornées. Tous les démarrages d'une vague sont rejoints avant EXEC; un échec empêche
les vagues suivantes et ferme les sessions déjà établies, sans rejeu automatique.
La résidence et les coefficients des experts restent identiques.

Les sorties ont une enveloppe JSON versionnée ; une entrée invalide produit une erreur
et un exit code non nul. `capabilities` indique explicitement que CUDA/PCIe/RDMA et
les workers cloud réels n'ont pas été qualifiés.

## Ce qui existe réellement

Le crate Rust contient un calcul MoE séquentiel et un calcul parallèle par rang CPU,
un registre global/local avec répliques, un dispatch compact, un collecteur strict,
une référence de décodage NVFP4 linéaire, un budget mémoire avec pic de chargement,
une topologie orientée, un comptage de trafic, un placement heuristique borné, un
partitionnement pipeline et un ordonnanceur analytique de dépendances.

Le collecteur refuse notamment doublons, résultats manquants, mauvais identifiants,
ancien epoch, valeurs non finies et expiration. Il ne change pas les experts ni leurs
coefficients pour dissimuler un échec. Ce n'est pas un protocole réseau exactly-once.

Le socle initial utilise des threads CPU. La continuation A1–A3 ci-dessous ajoute
des processus locaux réels, un codec QPN2 et une FFI SM70 ; le transport entre machines
et l'exécution GPU restent à qualifier.
Le planner n'est pas un optimum global ; le DP n'est optimal que pour ses stages ordonnés
fixes et ses coûts additifs. Les fixtures SYNTHETIC ne décrivent pas la topologie réelle
du propriétaire et ne prédisent pas des tokens/s.

## Utiliser LiteLLM sans inventer les alias

`agents/routes.example.json` laisse volontairement les quatre alias à null. Les familles
annoncées par le propriétaire sont Muse Spark Contributor1.3, DeepSeekV4.1, GLM5.3Flash,
et ChatGPT6Luna. Codex doit vérifier les noms réellement exposés par le gateway existant,
les contrats et les limites avant d'envoyer les canaries puis les vagues.

Dry-run local, sans clé ni requête réseau :

```sh
python3 scripts/vf_agents.py run --config agents/routes.example.json --queue agents/queue.example.json
```

Le fichier d'exemple contient huit revues indépendantes. Le prompt autorise des dizaines
de contributions par famille lorsque cela aide ; concurrence initiale de huit, montée
conditionnelle vers seize ou trente-deux. Pas de quota artificiel ni d'arbre infini.

Le runner fourni est un client de **propositions sans outils**, pas un agent Codex complet.
Ses réservations SQLite limitent les appels et les tokens de sortie par run. Elles ne
plafonnent pas seules les dollars, les tokens d'entrée/raisonnement ou les autres clients.
Les limites réelles du gateway doivent être vérifiées. Aucune clé et aucun alias inventé
ne sont incorporés. Aucune réponse modèle ne constitue une preuve d'exécution.
Voir `docs/AGENT_PLAYBOOK.md` pour les commandes et limites détaillées.

## Organisation

- `src/`, `tests/` : bibliothèque Rust, CLI et tests ; tests du runner dans `tests/python/`.
- `scripts/`, `fixtures/`, `agents/` : vérification, oracle, client de propositions et exemples.
- `docs/`, `handoff/`, `evidence/` : contrats, prochains lots, preuves et reçus.
- `reference/prior_qualification/` : recherche héritée, 42 entrées, pins et ancienne preuve CPU.

Les anciennes références sont historiques, pas 42 nouveaux audits. Aucun kernel tiers
n'est recopié automatiquement. La licence de publication du code nouveau reste à choisir
par le propriétaire ; les dépendances gardent leurs propres droits.

## Frontière de livraison

Restent à qualifier : kernels sur V100, CUDA graphs, mémoire/PCIe/inter-PC,
service de modèle et énergie système. Les tranches CPU A1–A4 sont détaillées ci-dessous,
leurs preuves courantes dans evidence/DELIVERY.json. Aucun débit, gain de performance,
test longue session ou coût financier d'agents vérifié n'est annoncé.

La qualification CPU distante a utilisé uniquement le dossier :
`/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c`.
Le pack est le handoff complet ; ce dossier distant peut ne contenir que le build et
ses preuves. Aucun service GPU, pilote, BIOS/VBIOS ou projet voisin n'a été modifié.

## Continuation A4 — placement analytique borné

Le planner compare désormais trois graphes complets pour chaque demande isolée :
baseline, regroupement et politique historique fondée sur les octets. Le champ
optionnel `wire_policy` accepte `baseline`, `grouped` ou `adaptive` (défaut).
L'adaptatif retient le plus petit temps du scheduler ; la baseline gagne les égalités.
Ce choix compare trois candidats, pas toutes les combinaisons possibles.

La sortie `demand_comparisons` conserve leurs temps, octets, nombres de messages et
le candidat choisi. Les sorties demeurent ANALYTICAL, y compris lorsqu'un coût
importé porte une référence de mesure. Aucun score ne devient des tokens/s.

Les experts et liens acceptent `cost_provenance` :
`{"origin":"assumed|synthetic|imported_measurement","source":"référence"}`.
Sans ce champ, une ancienne entrée reste `assumed`. Une mesure importée exige une
référence non vide ; l'étiquette et la référence ne certifient pas le contenu d'une
mesure. Les fixtures synthétiques du pack sont explicitement étiquetées.

`max_work_units` est optionnel, vaut 1 000 000 par défaut et accepte 1..=1 000 000.
Le compteur partage une seule réserve pour la validation, le placement initial,
les liens, les copies de candidats, les trois graphes, les scans du scheduler et
les diagnostics finaux. Une unité réserve une inspection d'entrée ou de tâche ;
ce n'est pas une borne de temps mural, de mémoire en octets ou de tokens.
`WORK_BUDGET_EXCEEDED` produit une erreur sans placement partiel.
`max_iterations` continue de limiter les évaluations de candidats ; `termination`
distingue sa limite d'un arrêt de la recherche locale.

Les anciens JSON restent lisibles. Les constructeurs Rust de `ExpertProfile`,
`Link` et `PlacementProblem` doivent fournir les nouveaux champs (voir les tests).
Les identifiants/ressources du scheduler sont bornés à 256 octets, les IDs de liens
à 128, les hôtes à 256 et les notes/références à 4096 pour borner leurs copies.

`scripts/verify.py` inclut maintenant `scripts/planner_crosscheck.py`. Ce second
oracle utilise des formules fermées indépendantes pour deux experts distants de
coût égal, avec transferts asymétriques et bus partagé facultatif. Il couvre 198
cas CPU analytiques et ne qualifie ni un checkpoint, ni un GPU, ni une file de
requêtes concurrentes.

## Continuation A1–A3 — interface quantifiée et processus

Le Skinny lu est le dépôt gilby/volta-nvfp4, commit
b1a2181702e609dc92591343e46c63d1324ade43, acquis dans le dossier isolé du chantier.
L'adaptateur `adapters/skinny_moe.py` conserve les plans compressés QPN2 et appelle
l'extension réelle. Ses tests injectent explicitement un oracle CPU ; ils ne lancent
pas le kernel. Pour un rank partitionné, `selected_mask` est obligatoire.
`bind_rank_dispatch` construit ce masque depuis le RankDispatch Rust et vérifie le
Stamp courant, l'ordre physique, les IDs et les coefficients. Ne pas conserver un
masque de batch sur la couche. Le combine utilise FP32 et refuse ses débordements.
La compaction dynamique et les scans de validation ne sont pas qualifiés pour les
graphes CUDA ; leur coût reste à mesurer.

`vf routing` consomme seulement des métadonnées et des routes. L'ordre physique des
poids est explicite, indépendant des IDs logiques clairsemés. `vf rank-run` lance de
vrais enfants par pipes privés et réutilise le collecteur strict. Le modèle dense
transporté par cette commande est un oracle CPU borné, pas le service quantifié.
Les reçus distinguent NotStarted prouvé, Started, Unknown et Completed. Admission
sans réponse ne signifie pas absence d'exécution. Une erreur supprime toute sortie,
conserve les reçus, tue/reap les enfants et retourne exit 2. Une reprise explicite
utilise un nouvel attempt ; l'ancienne réponse est refusée. La garde de replay est
locale au coordinateur, sans journal durable ni garantie exactly-once.

`vf qpn2-pack` et `qpn2-unpack` permutent les codes/scales sans décompresser tous les
experts. `qpn2-dequant` est un oracle CPU borné. Le vrai checkpoint NVIDIA Qwen3.6
35B A3B NVFP4 est épinglé à 1355db6a052410cfd62085d94b58866fd0f2c3c5.
Un seul expert/couche est échantillonné : crops 64×64 gate/up/down, aucune inférence
du modèle complet. Le prépack Torch amont, l'export safetensors indépendant et les
décodeurs Rust concordent sur 12 288 valeurs à atol=rtol=0.

La FFI `cuda/qpn2_moe_ffi.h` emprunte des pointeurs de plans compressés et les buffers
d'activation/sortie. Le validateur CPU contrôle dimensions, couverture unique,
padding et non-chevauchement de la sortie. Les miroirs host doivent correspondre
exactement aux métadonnées device. L'appel est asynchrone : OK signifie enqueue,
jamais fin de calcul. Le cœur SM70 conserve les opérateurs du donneur et ses droits.
Compilation CUDA12.8.93/GCC14.3 isolée, six déclarations noexcept de compatibilité
libc tracées ; ni CUDA13 ni installation système. Cette preuve est CUDA_COMPILED,
sans qualification GPU ou extension Torch complète.

Depuis `source/`, dans ce chantier déjà configuré :

```sh
VF_ROOT=/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c
export RUSTUP_HOME="$VF_ROOT/.tools/rustup"
export CARGO_HOME="$VF_ROOT/.tools/cargo"
export CARGO_TARGET_DIR="$VF_ROOT/target"
export RUSTUP_TOOLCHAIN=1.90.0
export PATH="$CARGO_HOME/bin:$PATH"
"$VF_ROOT/.tools/adapter-cpu/bin/python" scripts/check_integrity.py
"$VF_ROOT/.tools/adapter-cpu/bin/python" scripts/verify.py --run-id cpu-new-unique-id
bash cuda/build_sm70.sh "$VF_ROOT"
"$VF_ROOT/target/release/vf" rank-run fixtures/process_cpu.json
```

La vérification complète exige maintenant Torch CPU, safetensors, packaging, NumPy,
Ruff et un compilateur C, en plus de Rust. `requirements-cpu.txt` épingle les outils ;
Torch 2.10.0+cpu se résout séparément via son index CPU. Les caches/installations
restent sous Naos. `--python-only` reste une preuve partielle. Les steps supplémentaires
testent l'adaptateur, l'échantillon réel, lint/format Python et la FFI sous ASan/UBSan.

La baseline corrigée sert d'oracle de valeurs et de coefficients : ce lot apporte
la correction du partitionnement et une frontière exécutable, aucun speedup mesuré.
Les statuts finaux et hashes sont dans `evidence/continuation_a123/phase.json` et
`evidence/DELIVERY.json`. H1/H2/H3 restent ouverts : CUDA graphs, précision GPU,
mémoire réellement disponible, PCIe/inter-PC, charge longue et énergie.

## Continuation A5 — collecte sans modèle dense

`Collector::from_routing` accepte un `RoutingProblem` validé et utilise les mêmes
`Completion`, contrôles Stamp/replica/identité, état terminal et ordre de somme que
le collecteur dense. La commande `vf collect-routing` reçoit
`{"problem":<RoutingProblem>,"completions":[<Completion>],"expired":false}`.
Ni matrices d'experts ni activations ne sont requises au coordinateur.

`Completion.value` est la sortie **non pondérée** de l'expert. L'entrée nouvelle
`adapters.skinny_moe.moe_apply_local` renvoie les positions originales aplaties et
les lignes h2 compactes FP16 du même provider Skinny. Elle conserve les plans
compressés et les contrôles de l'adaptateur ; aucune fallback CPU n'est activée.
L'appelant associe ces lignes au Stamp courant déjà validé et aux IDs logiques du
RankDispatch. `moe_apply` garde son combine local existant, déjà pondéré : ne pas
l'envoyer au collecteur comme une sortie brute.

Le collecteur metadata conserve le coefficient f64 exact du plan, l'applique une
fois et somme dans l'ordre original des slots, quelle que soit l'arrivée. Son
oracle de reçus est f64 ; cela ne certifie pas les sommes GPU FP32 d'un service.
La nouvelle limite `tokens × topk × hidden ≤ 16 777 216` borne les valeurs f64
reçues à 128 MiB, avant construction du dispatch. Ce n'est pas une limite de RSS
totale : métadonnées, conteneurs, entrées sérialisées et sortie s'ajoutent.
Le constructeur dense existant conserve sa validation et ses limites.

Les tests confrontent la nouvelle voie à l'oracle dense corrigé et exercent
répliques, ordre physique inversé, rank vide, slot de poids nul obligatoire,
coefficients au-delà de la précision float32, omissions/duplications/ancien Stamp,
débordement et somme sensible à l'ordre. Le contrôle de sensibilité détecte la
double pondération. Tests numériques du pont exécutés avec un oracle CPU explicite ;
aucun kernel V100 ni performance GPU n'est qualifié.
## Continuation A6 — buffers compressés possédés

La bibliothèque Rust expose `resident::pack(ResidentInput)` et
`ResidentStack::bind_dispatch(&RankDispatch)`. La pile possède les Vec codes/scales
QPN2 et facteurs F32 ; ses champs sont privés, ses slices accessibles en lecture.
Chaque expert contient gate puis up dans le buffer13, down dans le buffer2.
Les facteurs gate/up distincts sont répétés par tile32. Aucun expert n'est
décompressé pour construire cette pile. Clone duplique le payload compressé.

L'entrée précise model_revision, layer, rank, registry, resident_order et les
experts id/local avec trois PackedRows gate/up/down. Validation avant permutation :
copie rank-local exacte, ordre couvrant tous les experts, formes uniformes et
compatibles, alignement rows32/columns64, échelles unsigned finite E4M3 et facteurs
positifs exactement représentables en F32. La limite est 1 048 576 éléments par
projection et 128 MiB de payload final codes/scales/facteurs. Entrées, copies,
temporaires et métadonnées s'ajoutent ; ce n'est pas un plafond de RSS.

La liaison contrôle modèle/couche/epoch/rank/ordre physique et H pour une pile
non vide ; elle rejette les slots étrangers ou dupliqués. Run/plan/batch peuvent
changer pour réutiliser les mêmes poids statiques. Le routage global décide
toujours du propriétaire. Une pile vide garde une forme zéro et n'admet que des
slots vides ; l'exécuteur doit sauter le kernel.

CLI : `vf resident-pack` reçoit ResidentInput ; `vf resident-bind` reçoit
`{"weights":<ResidentInput>,"dispatch":<RankDispatch>}`. Ces commandes valident
des buffers CPU possédés ; aucun upload, allocation device ou kernel n'est exécuté.
L'oracle indépendant est directement exécutable dans le chantier configuré :

```sh
"$VF_ROOT/.tools/adapter-cpu/bin/python" scripts/resident_crosscheck.py --binary "$VF_ROOT/target/release/vf"
"$VF_ROOT/target/release/vf" collect-routing fixtures/collection_metadata.json
```

L'oracle compare 27 648 octets codes/scales et 24 facteurs F32 bit à bit au
prépack Torch amont, avec deux ordres physiques, 12 roundtrips de projections,
six rejets de liaison et un rank vide. Les IDs10/99 sont synthétiques et portent
les crops du vrai expert0 ; le facteur up de ID10 est doublé comme contrôle
déclaré. Ce n'est ni un chargement du modèle complet ni une mesure de performance.

Qualification finale A5/A6 : 147 tests Rust distincts dans chacun des profils,
82 tests Python, 923 cas core, 198 cas planner, 12 288 valeurs du crop exactes,
51 assertions FFI sous ASan/UBSan. Preuves et erreurs de harness conservées dans
`evidence/continuation_a56/` et `evidence/runs/phase-a56-final-20261008/`.
H1/H2/H3 restent ouverts ; commande matérielle préparée `handoff/hardware_canary.sh`
avec V100 réservée et interpréteur CUDA12/SM70 existant.


## Continuation A7 — préparation host du lancement FFI

`launch::prepare(&ResidentStack, &RankDispatch, &Stamp, &Options)` lie le Stamp
complet courant et la pile statique. Le résultat emprunte les poids immuables,
possède les métadonnées compactes et ne peut pas être désérialisé sans validation.
Aucune copie des poids ni allocation d'activations/GPU n'est exécutée.

Les lignes compactes sont triées par token/slot original. `source_tokens` prépare
le gather, `completion_slots` conserve global/local/rank et coefficient f64 exact.
Les blocs de huit indices compacts sont groupés par expert physique. Le sentinel
de padding vaut le nombre de lignes compactes ; les coefficients nuls restent
des slots obligatoires. La même métadonnée sert aux deux projections, in_div=1,
nacc=1 et splitk explicite dans les configurations compilées, divisible par K/16.
Une absence de slots donne skip=true, métadonnées vides et projections nulles.

`max_workspace_bytes` est requis, positif et au plus 128 MiB. La somme compte
compact_x FP16, gate/up h1 FP16, activation FP16, down h2 FP16 et un jeu de
métadonnées device : 2*S*(K13+N13+K2+N2)+36*blocks+4 octets.
Elle est vérifiée avant construction des tableaux de lancement. Ce plafond
décrit ces buffers nommés ; il exclut poids résidents, activations sources,
miroirs host, copies, temporaires supplémentaires de SwiGLU, combine FP32,
transports et pic réel d'un exécuteur. Les capacités ne garantissent ni résidence
GPU ni réussite d'une allocation device. Aucun débit ou optimisation mesurée.

`vf resident-launch` reçoit :
`{"weights":<ResidentInput>,"dispatch":<RankDispatch>,"expected_stamp":<Stamp>,
"options":{"splitk13":4,"splitk2":4,"max_workspace_bytes":134217728}}`.
Le coordinateur fournit le Stamp actuel ; cette égalité ne constitue pas une
authentification ou une vérification indépendante d'un plan réseau.

La suite `scripts/verify.py` compile le vrai validateur C en bibliothèque CPU,
puis `scripts/launch_crosscheck.py` confronte les métadonnées à la sémantique
indépendante des routes et à l'ABI208. Sur les 66 cas, 521 appels C et 455 rejets
CLI sont exécutés. Des IDs/routes synthétiques portent les crops réels de
l'expert0 ; des adresses opaques alignées servent seulement à la validation CPU.
Elles ne sont pas présentées comme des allocations device accessibles.

Un contre-exemple change un expert physique vers un autre ID valide : la
couverture C reste correcte, mais l'oracle de routage le rejette. Le constructeur
Rust, ses champs privés et le dispatch courant doivent préserver cette
association. Les uploads host/device, leur égalité et le maintien des buffers
sur le même stream jusqu'à synchronisation appartiennent toujours à H1.

Dans le chantier configuré, une nouvelle qualification complète s'exécute :
```sh
"$VF_ROOT/.tools/adapter-cpu/bin/python" scripts/verify.py --run-id cpu-a7-new-unique-id
```
Preuves A7 : `evidence/continuation_a7/phase.json` et
`evidence/runs/phase-launch-final-20261008/summary.json`.

## Continuation A8 — sorties compactes FP16 vers reçus

`completion::from_output(&Prepared, &RawOutput)` utilise le plan préparé validé,
pas un snapshot JSON désérialisé. Le Stamp complet, le rank, le nombre de lignes,
la largeur cachée et la longueur exacte sont contrôlés. Aucune ligne de padding
n'est admise dans la sortie compacte. La largeur est conservée dans Prepared,
y compris pour un rank vide ; un paquet vide conserve ses dimensions et identité.

Les u16 sont les bits IEEE binary16, en ordre row-major token/slot compact :
celui de `completion_slots`, pas l'ordre expert-major des blocs de calcul.
Le tableau numérique JSON n'a pas d'endianness ; une lecture binaire de staging
doit décoder les mots little-endian. Tous les FP16 finis sont exactement
convertis en f64, y compris subnormaux et zéro signé. NaN et infinis sont refusés
avant allocation des reçus, même en dernière position. La fonction retourne
le paquet entier ou une erreur ; elle n'alimente pas progressivement un sink.

Les identifiants global/local/token/slot viennent du plan. Aucune pondération
n'est appliquée à `Completion.value` : le collecteur garde l'autorité des
coefficients f64 et l'ordre de somme. Une deuxième conversion est possible ;
le collecteur refuse ses doublons. Ce n'est pas un protocole exactly-once,
une authentification, un journal durable ou une preuve d'exécution.

La borne S*H ≤ 16 777 216 limite les seuls scalaires f64 des reçus à 128 MiB.
Elle précède scan et allocation de sortie. Stamps, Vec, entrées FP16, poids,
copies et sérialisation s'ajoutent ; pas de borne de RSS. Les réservations des
Vec retournent ALLOCATION_FAILED si la réservation échoue. Le CLI conserve
sa borne d'entrée JSON de 16 MiB.

`vf resident-completions` reçoit `{"launch":<entrée resident-launch>,
"output":{"stamp":<Stamp>,"rank":2,"rows":S,"columns":H,"fp16_bits":[...]}}`
et produit les `completions` destinées à `vf collect-routing`.
Cette commande reconstruit les poids/plan pour l'exemple CLI ; la bibliothèque
réutilise le même ResidentStack/Prepared entre préparation et conversion.

L'exécuteur doit prouver la fin du GPU, le maintien des buffers et la provenance
des bits avant cette conversion. Une égalité de Stamp ne peut détecter une
permutation arbitraire ou une corruption de valeurs finies accompagnée de
métadonnées mensongères. Le kernel doit écrire chaque ligne à son index compact.
Ces invariants d'upload, d'ordonnancement et de lecture restent à qualifier en H1.
CPU_OUTPUT_REPACK_ONLY ne signifie pas CUDA_EXECUTED.

L'oracle indépendant NumPy compare les bits f64 issus des 63 488 encodages FP16
finis, sans tolérance. Les 2 048 encodages non finis sont tous rejetés dans le
test exhaustif Rust ; cinq représentants sont aussi injectés dans des paquets.
Le chemin deux ranks/répliques/ordre inversé/coefficient nul/rank vide produit
six reçus et concorde exactement avec une somme NumPy indépendante. Le contrôle
de sensibilité détecte la double pondération ; 24 entrées CLI adverses sont rejetées.
Les activations du contrôle sont explicitement synthétiques ; les crops du vrai
checkpoint restent compressés et aucun GEMM n'est simulé ou annoncé exécuté.

`scripts/verify.py` intègre ce contrôle. Depuis le chantier configuré :
```sh
"$VF_ROOT/.tools/adapter-cpu/bin/python" scripts/verify.py --run-id cpu-a8-new-unique-id
```
Preuves A8 : `evidence/continuation_a8/phase.json` et
`evidence/runs/phase-completion-final-20261008/summary.json`.

## Continuation A9 — lancement matériel relié au cœur Rust et reçu persistant

Le canary H1 prépare deux modèles MoE minuscules via le vrai routing Rust,
resident-pack, resident-launch, resident-completions et collect-routing. Il
upload les poids QPN2 compressés, préserve les coefficients et saute le rang vide.
Les deux GEMM et SwiGLU gardent leurs buffers jusqu'à synchronisation explicite.
Cette tranche est un canary froid avec subprocess/upload, pas un service ni un benchmark.

Le reçu matériel est créé exclusivement avant tout accès CUDA, puis persisté avec
fsync du fichier et du répertoire. ENQUEUE_PENDING est écrit avant l'appel C ;
ENQUEUED signifie seulement lancement accepté. Une synchronisation échouée garde
les références et l'issue incertaine. Un reçu existant, incomplet compris, bloque
le nouveau lancement. Le nonce protège les acteurs qui respectent ce protocole,
sans authentification ni garantie exactly-once. Une absence de réponse ne prouve
pas une absence d'exécution. FAIL/BLOCKED ne deviennent pas PASS au cleanup.

Treize tests nouveaux couvrent deux processus concurrents réels, la mort après
persistance du pending, les erreurs de synchronisation, les identités périmées,
les propriétaires alternatifs et la collecte globale. L'oracle CPU est injecté
uniquement dans les tests ; le main matériel garde le provider C réel.
Qualification complète : 163 tests Rust par profil et 95 tests Python. Traces rouges,
revues indépendantes et reçus dans evidence/continuation_a910. Aucun GPU exécuté.

## Continuation A10 — expert réel complet, export indépendant hors ligne

La fixture safetensors contient les neuf tenseurs complets de couche 0/expert 0
du même checkpoint NVIDIA épinglé, H=2048 et I=512. Codes/scales restent compressés ;
les trois facteurs weight_scale_2 F32 little-endian sont directs. L'oracle compare
les 1 769 856 octets de payload résident et 96 facteurs par tile au prépack Torch
amont, avec égalité exacte, puis vérifie les préfixes contre les crops antérieurs.
Aucune matrice flottante de poids n'est décompressée. Le modèle complet et la
précision GEMM GPU restent non qualifiés.

Acquisition tracée : 1 931 563 octets de corps HTTP en dix ranges 206, sans shard
entier ni retry. Fixture : 1 770 932 octets, SHA256
7ede48f5bb49caa171eaf03c691b8d78d11e8644c465e4d008f8bcedf52c1c3a.
Provenance et licence Apache-2.0 accompagnent la fixture ; le corpus privé original
reste conservé. Reproduction hors ligne depuis source avec l'environnement Naos :

~~~sh
"$VF_ROOT/.tools/adapter-cpu/bin/python" scripts/full_expert_crosscheck.py --binary "$VF_ROOT/target/release/vf"
~~~

Le bénéfice démontré est la correction de la frontière et du format réel complet,
pas un gain de débit. Appels payants gateway : zéro ; coûts USD des agents natifs
inconnus, caps inchangés. H1/H2/H3 restent OPEN.

## Continuation A11 — import safetensors natif sélectif

checkpoint::read et checkpoint::load lisent le header borné puis seulement les
neuf tenseurs de chaque expert sélectionné. Le parseur safetensors 0.8.0 est
réutilisé sans modification, avec Cargo.lock, licence et provenance conservées.
Les contrôles portent sur offsets/EOF exacts, noms et labels uniques, U8/F8_E4M3,
formes SwiGLU, scalaire F32 little-endian direct, registre et ordre physique.
resident::pack produit les mêmes buffers QPN2 détenus. Aucun poids flottant ni
grande liste JSON de poids n'est requis par l'API Rust native.

Depuis source, avec le binaire déjà compilé :
~~~sh
"$VF_ROOT/target/release/vf" resident-import fixtures/resident_import_qwen_expert0.json
~~~
Le chemin de fichier est relatif au cwd du processus. La sortie CLI expose un
snapshot JSON compressé pour inspection ; un intégrateur Rust conserve le
ResidentStack binaire et emprunte ses buffers aux interfaces existantes.

L'oracle sur l'expert complet est exact. Un vrai fichier sparse de 270 206 484
octets exige 1 771 028 octets de lectures de payload/header dans ce test ; son
tenseur non sélectionné de 256 MiB n'est pas lu. Ce compteur de Read n'est pas
une mesure de trafic disque, de RSS, de latence ou de débit. Onze tests Rust
nouveaux donnent 174 tests distincts debug/release ; les 95 tests Python restent
verts. La course chemin régulier vers FIFO est traitée par O_NONBLOCK sous Unix,
avec un vrai reproducer arrêté/reap et un test Rust. Aucun unsafe ajouté au cœur.

Le fichier et son chemin doivent rester quiescents. Les labels ne sont pas une
authentification ; len/mtime ne protègent pas contre un writer hostile restaurant
les métadonnées. source_expert désigne explicitement l'indice du checkpoint ;
id/local désignent les identités logiques du registre. Tout remapping ou réemploi
de source_expert est la politique déclarée du manifeste appelant, pas une
validation automatique de la sémantique du modèle. Le request porte un seul shard
et les noms Qwen NVFP4 ; aucun loader multi-shard ou service GPU n'est annoncé.
Les limites couvrent header/payload et travail de packing, pas le pic RSS.
Preuves et deux revues dans evidence/continuation_a11 ; H1/H2/H3 restent OPEN.

## Continuation A12 — plans Rust reliés au vrai validateur C

La feature optionnelle c-abi-validation ajoute Prepared::validate_c(current,
gate_up, down). Le cœur conserve forbid(unsafe_code). Une petite crate privée
contient l'unique appel unsafe synchrone, vers le validateur C existant non
modifié. Elle dérive les longueurs HOST des slices empruntées, garde la structure
ABI privée et ne déréférence aucun pointeur DEVICE. Aucun launch CUDA ni backend
d'inférence n'est ajouté.

Une sonde C native contrôle ABI1, sizeof208, align8, ainsi que offset, taille et
alignement des 31 champs ; les assertions Rust correspondantes sont évaluées à
compilation. Le build canonique prend uniquement le header/validateur du projet,
sans override de source ; le Cargo.lock racine reste l'unique verrou du workspace.
Le mode par défaut du cœur reste indépendant de cette feature/compilerC.

~~~sh
cargo test --locked --workspace --all-targets --features c-abi-validation
~~~

La liaison utilise le vrai expert importé et les métadonnées Prepared immuables,
refuse Stamp périmé/capacité insuffisante/alias et saute le rang vide. Deux appels C
individuels peuvent admettre une sortie gate/up chevauchant les scales down :
le reproducer rouge est conservé et le contrôle de couche rejette ce cas.
Les quatre espaces x/h1/activation/h2 sont distincts et ne chevauchent les poids
persistants et métadonnées d'aucune des deux projections. Aucune réutilisation ou
activation in-place n'est autorisée sans qualification séparée.

La vérification complète passe 174 tests du cœur par défaut et 188 tests distincts
du workspace avec feature, dans chacun des profils debug/release, ainsi que 95
tests Python. On ne somme pas ces exécutions pour inventer plus de tests uniques.
Validation CPU de déclarations ne prouve pas accès GPU, contenus des buffers,
égalité des miroirs, absence d'alias physique via mappings distincts ou completion.
Le caller conserve ses obligations GPU ; H1/H2/H3 restent OPEN. Les preuves,
revue indépendante, régressions et reçus sont dans evidence/continuation_a12.


## Phase A13 qualifiée (8 octobre 2026)

Les corrections conjointes Codex/Claude du DP, des échelles NVFP4 unsigned et des
namespaces de jobs sont intégrées. La validation des échelles utilise une réduction
scalaire ; son scratch CUDA et son coût GPU restent non mesurés. Le code n'a pas
accédé aux V100.

Preuves actuelles : [livraison A13](evidence/continuation_a13/DELIVERY.md),
[gate CPU final](evidence/runs/phase-a13-qualified-20261008/summary.json),
[handoff](handoff/TASKS.json). 197 tests cœur / 211 workspace FFI par profil
debug/release, 101 tests Python ; oracle A13 de 473 cas. H1–H3 restent ouverts.

## Qualification CPU A16 — 9 octobre 2026

D01–D03 refusent les valeurs réellement converties/transmises non finies avant
le provider concerné et les doublons logiques même masqués. Les permutations,
répliques, coefficients appliqués une fois et gardes du canary sont testés.
D04 qualifie uniquement l'admission CPU de l'ordre d'opérations FP16 du backend
SM70; les poids Rust restent propriétaires/immutables et les Tensor Python
mutables sont revalidés, sans cache fondé sur un pointeur. Les pointeurs DEVICE
opaques du C ne constituent pas une attestation de leurs valeurs.

D05 conserve une alternative de lien finie lorsqu'un autre coût déborde.
D06 calcule la moyenne des coûts de graphe binary64 par somme dyadique exacte,
puis un seul arrondi au plus proche, ties-to-even. Fréquences finies non négatives,
au moins une positive pour des demandes non vides; demandes vides: score0.
Une moyenne exacte positive arrondie à zéro est refusée explicitement. Les
demandes de fréquence0 gardent leurs validations structurelles. La limite de
1024 demandes et le budget de recherche restent actifs. Ce score analytique
n'est ni un débit ni une preuve d'optimalité du placement.

Preuves actuelles: evidence/continuation_a16/phase.json et
evidence/runs/phase-a16-v4-final-20261009/summary.json. Rust1.90.0/Cargo.lock,
debug/release/workspace FFI, lint, oracles et ASan/UBSan C sont requis.
resident-inspect fixtures/resident_import_qwen_expert0.json fournit un résumé
borné sans construire le grand JSON; resident-import garde l'export complet
explicitement demandé, utile aux consommateurs existants. Sur un expert réel,
la sortie complète est bit-identique et son RSS CPU mesuré diminue; ces chiffres
ne qualifient pas la mémoire d'un modèle complet ni une V100.

Le paquet delivery/VoltaFabric-H1-A16R2-20261009.tar.gz dans la racine de chantier
contient sources, binaire/ABI, SM70, runtime CUDA et licences. Après extraction:
python3 source/scripts/h1_preflight.py --bundle . --out ../h1-static.json
Le binaire fourni exige Linux x86_64/glibc>=2.39. ARM/ancienne libc: reconstruction
native et nouveau préflight; crates non vendorisées. La bibliothèque est
CUDA_COMPILED seulement. Après réservation et docs/HARDWARE_GATE.md:
VF_GPU_PYTHON=/existing/verified/python VF_GPU_UUID=GPU-reserved bash run_h1.sh
L'interpréteur CUDA12/SM70 et le vrai UUID restent à fournir. Un canary vert
ne ferme pas H1 complet, H2 inter-GPU/inter-PC ou H3 service/énergie.

## Continuation A164 — collecte directe des lignes FP16

`vf collect-fp16` reçoit `{problem, expected_attempt, outputs:[...]}` où chaque
sortie porte son Stamp, son rank, son essai, ses dimensions, les identités
token/slot/expert/local et les bits u16 row-major. Les coefficients restent
uniquement dans le problème de routage canonique. La commande lie l'essai,
valide les identités et les valeurs finies, puis remet les bits directement au
collecteur ; aucun vecteur de reçus f64 par slot ni poids résident n'est construit.
Le collecteur restaure l'ordre token/slot et applique les coefficients une seule
fois lors de `finish`. Une sortie dupliquée, périmée ou incomplète ne produit pas
de résultat.

Le test réutilise l'owner persistant Python avec son fournisseur-oracle CPU,
fait deux EXEC sur les mêmes poids, compare le résultat direct au chemin
historique f64 et à un oracle indépendant binary16, et rejette les identités
fausses, les mauvais epochs/essais, NaN/Inf,
les doublons et les lots incomplets. `fp16_payload_bytes` compte seulement les
bits conservés en mémoire Rust : 2 octets par valeur contre 8 pour les reçus f64
intermédiaires. À la clôture d'A164, le protocole persistant de session restait à intégrer.

## Continuation A165 — réponses FP16 sur session persistante

La session Rust annonce la capacité optionnelle
`cpu_completion_batch_fp16_raw_v1`. Quand `CompressedSeriesInput.fp16_completions` est activé,
le coordinateur l'exige avant tout EXEC et demande explicitement le format; le worker renvoie
les sorties non pondérées en bits IEEE binary16 little-endian avec un en-tête JSON borné.
Les conversions RNE, les identités, la forme, la taille totale et toutes les valeurs finies
sont vérifiées avant l'émission du premier frame. Une réponse mal formée, tronquée ou non
finie échoue sans résultat de batch; aucune conversion de secours en f64 ni rejeu n'est
ajouté. Si une valeur calculée est finie mais hors de la plage binary16, le worker émet
`Failed(FP16_OVERFLOW)` sans frame de complétion partielle et reste prêt pour le prochain
EXEC; une erreur d'E/S pendant l'émission reste potentiellement partielle et inconnue.
Le collecteur conserve les bits u16 jusqu'à sa validation d'identité et applique ensuite
les mêmes coefficients canoniques une seule fois. Le chemin f64 reste le défaut.

Le gate CPU complet A165 r3 passe 41/41 étapes sous Rust 1.90.0 sur PC6 aarch64, dont les
tests debug/Release, ABI, oracles et validations existantes. Un test session avec deux
workers locaux, six tokens et un petit artefact QPN2 synthétique compare neuf lots (deux
warmups) à l'oracle Python indépendant `struct` binary16; la sortie pondérée est bit-identique.
Dans ce seul essai local, les réponses comptées par la session font 130 072 octets en f64 et
47 218 en FP16. Les p50 observés (12 531,857 µs et 7 021,772 µs) et le RSS par worker
(3 880/3 884 KiB et 3 872/3 884 KiB) sont des diagnostics d'un petit essai, non une mesure
de performance ou de mémoire système. Ces octets n'incluent ni l'encapsulation SSH ni le
transport physique. A165 ne qualifie ni le vrai checkpoint DeepSeek, ni CUDA/V100, PCIe,
RDMA/GPUDirect, ni PC1↔PC6; H1/H2 restent ouverts.

Within one `session-run-mxfp4` call, repeated identical `artifact_path` strings
refer to one admitted immutable file snapshot. Its compressed backing is shared
across ranks; each binding/digest/expert mapping is checked independently. The
128-MiB client-owner budget counts those distinct allocations. Distinct paths
(including aliases) are read and validated independently, even with the same
declared digest. A later call always reopens and validates its files. Mutating or
removing a path after admission does not change the already admitted snapshot.

New unique paths in `session-run-mxfp4` are read under the smaller of the 64 MiB
per-artifact limit and the remaining 128 MiB compressed-owner budget. One sentinel
byte detects excess input. An exhausted budget is rejected before opening another
path; existing shared snapshots remain eligible without another allocation/read.
This bounds compressed input bytes, not total RSS, allocator overhead, or the time
spent opening/reading a nonregular source while a positive budget remains.
