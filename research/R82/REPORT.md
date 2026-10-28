# VoltaFabric — audit approfondi R82 du 27 octobre 2026

## Verdict et définition de « utilisable »

**VoltaFabric dispose d'un socle de calcul et de protocoles substantiel, ainsi que de preuves matérielles sur des experts GPU. La livraison examinée ne permet pas encore de charger un modèle complet, recevoir un prompt et générer une réponse.** La vitesse relative face à vLLM, llama.cpp et Strata est **inconnue**, faute de benchmark comparable.

Hypothèse d'usage retenue, en l'absence de précision contraire : un modèle complet qui répond de façon fiable sur les machines du propriétaire. Pour cet objectif, l'indice de préparation proposé est **25/100**, avec une plage de jugement **15–35/100**. Ce n'est ni une mesure de temps restant, ni un pourcentage de lignes écrites, ni une probabilité de succès. Il serait injustifié d'annoncer90% parce que beaucoup de composants ou de tests passent. L'absence du chemin complet reste une porte bloquante, même si la somme pondérée vaut25.

Notes de l'auditeur : **code7,5/10**, **projet de R&D7/10**, **maturité du produit livré3/10**. Les notes sont des jugements explicités ci-dessous, pas des résultats automatiques d'un outil ou une moyenne des opinions des agents. Confiance moyenne sur la note globale, forte sur l'absence de qualification du modèle complet. Une revue indépendante a proposé une estimation encore plus faible pour le service final ; elle n'a pas été moyennée artificiellement avec les autres avis.

## Alerte nouvelle : la porte CPU complète fraîche ne passe pas

La suite existante a été exécutée une seule fois pour cet audit, sans modification de ses seuils ni de son code. Résultat **FAIL :10 étapes réussies,1 étape expirée,34 étapes non exécutées sur45 prévues**. L'étape `cargo test --workspace --release --all-targets --locked --features c-abi-validation` a dépassé sa limite600s. Python, format, Clippy, Rust debug/release, doctests, Clippy FFI et FFI debug avaient réussi. Aucun échec d'assertion numérique n'a été observé dans ces étapes terminées. Les sources sont restées stables. Les profils Rust debug et release ont chacun exécuté661 tests réussis et ignoré16 tests préexistants de profiling/multihôte ; FFI debug compte677 réussites et les mêmes16 ignorés. Ces profils se recouvrent et ne sont pas additionnés en1969 tests distincts. Python a exécuté166 tests. Aucune exclusion n'a été ajoutée pour cet audit.

Ce résultat ne démontre pas un bug numérique. Les timestamps montrent60 exécutables de tests produits pendant la phase expirée, le dernier à598,17s : une limite de compilation/LTO en mode un job est plausible. Le profil historique R59 utilisait deux jobs/deux threads, celui de l'audit un job/un thread. La cause exacte reste **INCONCLUSIVE** : le runner ne conserve pas `TimeoutExpired.stdout`, et son log d'échec perd la progression capturée. Cette insuffisance diagnostique est un finding concret dans `scripts/verify.py`. Tous les descendants du groupe d'audit sont absents ; aucun GPU ni modèle n'a été lancé.

La suite n'a pas été relancée pour obtenir du vert. Les45 anciennes portes réussies restent historiques et ne remplacent pas ce résultat frais. Une livraison doit isoler le coût compilation/exécution et retenir les logs partiels avant de revendiquer un gate complet actuel, sans camoufler le timeout ou changer les caps après coup.

## Périmètre, indépendance et limites

Le propriétaire a explicitement demandé ce nouvel audit général. Trois revues indépendantes ont couvert architecture/opérabilité, calculs/tests et comparaison des moteurs. L'intégrateur a recontrôlé les sources, les manifests, les mesures et lancé la suite CPU existante sur le canonique courant. Le travail de production est en lecture seule : aucune réparation de code, nouvelle dépendance, installation, exécution GPU, modèle ou rejeu d'une identité UNKNOWN.

- Canonique R59 : **370 fichiers fonctionnels**, manifeste SHA256 `41f179ce7d7076748e907e566182aa1d3298b4deb776cb52ecd0f694edba526d`, zéro différence au début de l'audit ; stabilité finale dans les reçus.
- **Dix manifests de preuves recontrôlés**,3971 entrées couvertes au total, zéro mismatch : R49/R50/R59/R67/R69/R75/R76/R77/R80/R81_CPU. Ce total inclut des copies historiques ; ce n'est pas3971 nouveaux tests ni fichiers uniques. Un hash prouve les octets, pas la justesse d'un verdict.
- Commande publique `vf capabilities` fraîchement exécutée : CLI CPU/références/contrats, `performance_claim=NONE`; les indicateurs CUDA/RDMA du binaire canonique restentfalse. Cela ne nie pas les essais GPU privés distincts.
- Revue approfondie des chemins critiques sélectionnés, pas lecture exhaustive ni preuve formelle de toutes les lignes/dépendances. Aucun défaut numérique bloquant n'a été établi dans le R9 inspecté. Cela ne signifie pas absence universelle de bugs ou de vulnérabilités.
- Le répertoire canonique courant n'est pas un checkout Git (`git rev-parse` retourne128). La traçabilité locale s'appuie sur les manifests/diffs/paquets ; cela ne prouve pas qu'aucun dépôt Git n'existe ailleurs.

Les notes détaillées et leurs sources sont dans [architecture](/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/project-deep-audit-r82-20261027/agents/architecture-audit.md), [calculs/tests](/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/project-deep-audit-r82-20261027/agents/numerics-tests-audit.md), [comparaison](/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/project-deep-audit-r82-20261027/agents/engine-comparison-audit.md), `FINDINGS.json`, `EVIDENCE_MATRIX.json`, `PERFORMANCE_FACTS.json` et `SCORECARD.json`.

## Où en est le produit, composant par composant ?

| Domaine | Ce qui est établi | Ce qui manque pour l'usage final |
|---|---|---|
| Formats/calculs CPU | Admission, formats comprimés, référence FP32/K32, identités/collecte ordonnée, oracles et régressions | Validation du modèle complet et de ses vraies activations, pas seulement projections/expert |
| Expert GPU résident | Vrais essais V100/RTX5080, poids comprimés chargés une fois, sorties exactes dans les corpus définis | Couverture complète des formes/activations et intégration modèle ; bornes explicites de laboratoire |
| Sessions persistantes | LOAD/EXEC/BYE, rebind, défauts ciblés, refus d'identités/slots, deux appareils hétérogènes et SSH réellement exercés | Service continu, supervision, files de requêtes, reprise durable et charge représentative |
| Nouveau candidat d'entrée compacte R76 |81920 scalaires natifs exacts, refus FTZ/sticky poison R79, paquet Rust CPU R80 | R81 n'a pas exécuté le GPU : staging/contrôleur/admission puis première Session acceptée restent ouverts |
| Diagnostic mémoire du candidat | R77 a observé de vraies sorties exactes et rapports CS0erreur/0fuite | Porte complète UNKNOWN : codes normaux perdus, jamais remplacés par une déduction depuis les résumés |
| RDMA | Contrats CPU et essais de transport bornés historiques | GPU host-staged RDMA complet et GPUDirect non qualifiés ; SSH GPU n'est pas preuve RDMA |
| Modèle complet | Métadonnées/imports, primitives d'experts, adaptateurs Skinny ciblés | Tokenizer, embeddings, attention/KV, normalisations/résiduels, routage calculé par le modèle, orchestration de toutes couches, tête de sortie, sampling et boucle de génération intégrés/qualifiés |
| Usage quotidien | Outils de recherche et composants utilisables par un opérateur expert dans leur périmètre | Installation actuelle unifiée, commande prompt→réponse, santé/supervision, multi-utilisateur, SLO, longues sessions et validation de qualité |

Les primitives ou adaptateurs externes ne sont pas ignorés : `adapters/skinny_moe.py` est une vraie frontière MoE avec validation et exports. Son existence n'établit pas la preuve d'une requête complète traversant un vrai modèle et revenant à l'utilisateur. `docs/ARCHITECTURE.md:5–8` indique lui-même que Skinny reste la cible d'intégration. La CLI publique `src/main.rs:529–543` expose un atelier JSON de contrats et de sessions, sans commande de génération/serveur modèle complet.

Les limites actuelles sont concrètes : au plus4 experts par worker, formes multiples de32 dans32..5120,48M MAC par expert (`gpu-worker/src/main.rs:91–110`) et100M MAC par EXEC/rang (`src/session/mxfp4.rs:30–31`). Un expert réel5120×2304 représente35 389 440 MAC pour ses trois projections ; trois opérations de cette taille font106 168 320 et dépassent ce dernier cap. Les bornes protègent le périmètre qualifié ; elles ne sont pas une architecture générale de gros batching déjà livrée et ne doivent pas être relevées pour améliorer un résultat.

## Qualité du code

| Axe | Note /10 | Raisons et limites |
|---|---:|---|
| Contrats numériques et chemin critique examiné |8,5|Ordre d'arrondi déclaré, vérifications finite/forme, nibbles/scales bornés, refus explicites ; pas d'écart concret trouvé dans le R9 inspecté |
| Tests/oracles et honnêteté des preuves |8|Oracles indépendants, régressions sensibles aux mutants, rouge/vert, échecs initiaux et UNKNOWN conservés ; les petits corpus ne couvrent pas un modèle |
| Architecture/maintenabilité |6,5|Frontières Rust/FFI explicites, composants cohérents ; protocole concentré dans un très gros module et intégration utilisateur dispersée |
| Sécurité pour artefacts locaux de confiance |7|Admission bornée, identités et inode/FD significatifs ; pas sandbox hostile, service multi-tenant ou audit sécurité exhaustif |
| Documentation et navigation de l'état courant |5|Contrats précis mais README initial, nombreuses étapes privées et champs historiques mélangés rendent la version courante difficile à retrouver |
| Paquets/déploiement/opérabilité |4|Paquets privés reproductibles et essais ciblés réels ; chemins spécifiques aux hôtes, supervision et installation utilisateur unifiée encore absentes |

**Note du code retenue :7,5/10, plage7–8.** Ce jugement porte sur le prototype et ses chemins critiques, sans certifier l'ensemble du dépôt. **La note produit3/10 vient surtout des fonctions utilisateur et des qualifications manquantes, pas d'un bug arithmétique nouvellement démontré.**

Points forts vérifiés : le R9 conserve la réduction K32 puis l'arbre de lanes, l'encodageE8M0 code0 non nul et les vérifications d'overflow ; ses tests comportent annulation, sous-normaux, nibbles/lignes/échelles/permutations. La frontière de session stage les sorties jusqu'à couverture exacte et invalide les tentatives fautives. Le worker compare SHA compilé, symbole mappé et inode ouvert, puis conserve ce FD et recontrôle ses métadonnées. Il s'arrête sur erreur inattendue plutôt que rejouer ou basculer CPU silencieusement.

Limites : le chargeur OS a déjà mappé/initialisé une DSO avant son contrôle dans `main`; la garde d'identité suppose un paquet de confiance et ne protège pas d'une DSO hostile. Le fichier `src/session.rs` compte8560 lignes, et une grosse partie du protocole/schémas/modes y est réunie. Le runner principal collecte353 fichiers mais exclut le worker natif (`Cargo.toml:46` et `scripts/verify.py:16–55`) ; une porte CPU verte ne vaut donc pas qualification GPU globale. Les deux gros handoffs font ensemble1 030 732 octets au début de l'audit : `status` courant, `state` précédent et anciens blocages matériels non datés dans leurs noms coexistent. C'est un risque de lecture opérationnelle, même si le lecteur discipliné peut consulter les sous-entrées récentes.

## Vitesse : mesures réelles et absence de classement inter-moteurs

**Aucun facteur « VoltaFabric est X fois plus rapide que vLLM/llama.cpp/Strata » n'est justifiable.** Il manque un modèle complet VoltaFabric et des exécutions upstream comparables sur même matériel, poids/quantification, contexte, charge et qualité. Aucune valeur comparative en tokens/s, TTFT ou TPOT n'est disponible. « Non mesuré » ne signifie ni zéro tokens/s ni vitesse égale.

| Lot | Résultat observé | Portée exacte |
|---|---|---|
| R49/R50 |Temps des projections CPU isolées réduit d'environ26,13–26,17%; accès intégré dans R50 | Gain de référence CPU ; pas débit GPU/modèle |
| R67 |Regrouper huit entrées en lots de2 : octets requêtes338720→334168, soit−1,34388% | Baisse de framing ; mêmes327680B d'activations et144 kernels ; durées uniques non promues |
| R69 |Corps d'entrée réseau327680→163840B, soit−50%; requêtes complètes334168→170376B, environ−49,01% | Les deux peers ont négocié BF16 ; H2D GPU327680B inchangé,40960 sorties finales exactes |
| R75 |Quantizer CPU5120 valeurs : médiane wall108,523→11,831µs, soit−89,10% dans la série fixe | Microbenchmark de producteur CPU, staging/packing inclus ; pas inférenceGPU |
| R76 |Arguments Runtime inputH2D163840→42240B, soit−74,21875%,81920 sorties natives exactes | Baisse des octets d'entrée demandés à Runtime ; ni bytes physiques PCIe, ni vitesse, ni Session complète |

R76 n'a réduit ni les poids chargés une fois ni les sortiesD2H. Sur ces huit forwards, les requêtes Runtime chaudes input+output+status passent de327744B à206144B, environ−37,10% ; poids initiaux compris,19128384→19006784B, environ−0,636%. Ces deux calculs sont de simples sommes d'octets du cas observé, pas des mesures de latence et pas des prédictions d'un modèle. Ils illustrent pourquoi−74% sur une composante ne signifie pas−74% de temps moteur.

La source active du matvec R76 (`native/frozen-r9/candidate.cuh`) décode les poids et accumule en FP32 par warp avec arrondis explicites. Les poids4bits ne suffisent donc pas à établir une utilisation native TensorCoreFP4 ni le débit théorique d'une RTX. Le projet a aussi une frontière Skinny distincte ; aucune généralisation à tous ses futurs kernels n'est faite ici.

La [documentation officielle vLLM](https://docs.vllm.ai/en/latest/benchmarking/cli/) distingue débit de tokens, TTFT et latence inter-token à des frontières définies. [`llama-bench`](https://github.com/ggml-org/llama.cpp/blob/6050b467c07a95c7b87debd4ddac2f8ba8d9b807/tools/llama-bench/README.md) sépare prompt processing/génération et exclut tokenisation/sampling. Le [protocole Strata épinglé](https://github.com/Niko1221/Strata/blob/4b9e05a2a806a3fe24f7b2560d98626c37dade35/docs/BENCHMARK_PROTOCOL.md) reste draft avec pilotes matériels encore demandés. Aucun de leurs exemples publiés n'est utilisé comme mesure de nos machines. Voir le rapport comparatif indépendant pour révisions, formats, coefficients et liens primaires.

## Calcul du pourcentage demandé

| Critère du prototype complet utilisable | Poids | Crédit actuel |
|---|---:|---:|
| Formats et contrats numériques/d'identité |10|9|
| Exécution GPU d'experts résidents |15|10|
| Modèle complet et génération |30|0|
| Déploiement/intégration multihôte |15|3|
| Fiabilité opérationnelle/reprise/longues sessions |15|2|
| Qualité/performance/acceptation bout en bout |15|1|
| **Total** |**100**|**25**|

Cette pondération donne beaucoup de poids au résultat réellement demandé à l'utilisateur final. Les fonctions manquantes ou simplement prédites ne reçoivent pas le crédit d'une livraison validée. Le dernier point reconnaît l'instrumentation et la méthode existantes, pas un benchmark de modèle déjà réalisé. La plage15–35 traduit la sensibilité au choix des poids et au périmètre, sans prétendre être un intervalle statistique.

Pour la recherche/exécution bornée d'experts, une partie importante est déjà utilisable aujourd'hui avec opérateur expert et environnements qualifiés. Pour un moteur de chat complet ou un service autonome, la réponse actuelle reste **pas prêt**. L'indice25/100 ne doit pas être transformé en date ou en75% de temps restant : l'intégration manquante peut concentrer les difficultés.

## Priorités de livraison proposées

1. **Écrire un critère d'acceptation utilisateur unique** : un modèle nommé, un checkpoint/hash, un prompt, une réponse générée, une commande, puis plusieurs contextes et erreurs bornés. Utiliser d'abord une taille compatible avec une machine réellement réservée.
2. **Finir la frontière actuelle R81 une seule fois avec ses portes**, pour savoir si le worker réellement livré accepte le candidat. Préserver R77UNKNOWN, ses identités consommées et ses limites ; aucun nominal memcheck rejoué pour de meilleurs logs.
3. **Intégrer cette frontière à un moteur complet existant** dans le périmètre Skinny initial plutôt que recréer tout un serveur/tokenizer/attention/KV. Démontrer une tranche verticale utile avant d'ouvrir de nouvelles variantes d'optimisation.
4. **Créer un paquet actuel et une page d'état courte** : version/source/native/architecture, commande d'installation/exécution, limites réelles, tests frais et tests historiques distincts. Garder les archives sans recopier tout le handoff dans la page utilisateur.
5. **Mesurer un cas commun** : même modèle logique et qualité, même GPU/hôte/driver, mêmes contextes/sampling/batch/cache. Séparer cold-load, préfill, decode, TTFT/TPOT p50/p95, débit sous charge, erreurs et mémoire. Si les quants/layouts diffèrent, publier une comparaison qualité/performance de paquets, pas un avantage du moteur seul.
6. **Fermer la stabilité de service** : sessions longues, concurrence/annulation, mémoire/pic de chargement, redémarrage, supervision et absence de fuite avec les vrais artefacts finaux. Aucun score de production sans ces critères.

Ces actions sont une recommandation issue de l'audit, pas une implémentation ou un benchmark déjà exécuté. Aucune date fiable de livraison ou victoire de vitesse ne peut être promise à partir des preuves actuelles.

## Vérification fraîche R82

**Résultat :FAIL_TIMEOUT,10 étapes0/1 expiration/34 non exécutées, aucune relance.** Le résultat final de la suite CPU courante et les limites de son domaine sont consignés dans `evidence/current-CPU-gate-operation.json` et `evidence/current-CPU-gate-assessment.json`. Les sources restent celles du manifeste R59. La commande publique `capabilities`, les trois revues, les dix relectures de manifests et les sources officielles complètent ces preuves. Aucun nouvel essai GPU ou modèle n'a été lancé pendant l'audit.
