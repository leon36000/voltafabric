# R651 — callback C++ sur une flotte locale de 48 sessions

## Résultat vérifié

Le harnais appelle le vrai PoolFleetOwner::create_from_fds et pool_cpu_by_layer des sources C++ de l'adaptateur VoltaFabric déjà présentes à R650. Le Python utilise le lanceur existant uniquement pour ouvrir les sockets AF_UNIX et démarrer les coordinateurs Rust. Le test a démarré 48 vf-strata-native-bridge, chacun avec son processus vf-rank-session CPU, puis a transmis deux batches à chacune des 48 couches via le callback C++ : 96 appels et 96 reçus locaux distincts.

Pour chaque couche, la sortie FP32 était finie et les deux batches ont produit des octets identiques avec le même résident synthétique. Le test a aussi envoyé un index de couche invalide, vérifié que la destination sentinelle restait intacte, puis vérifié que la flotte refusait tout appel ultérieur. PoolFleetOwner::shutdown_all et la fermeture des coordinateurs par le lanceur ont réussi; les processus ont tous disparu.

Le pack temporaire a une taille logique déclarée de 50 292 326 400 octets. Le harnais en borne l'allocation physique à 128 MiB; il ne contient qu'un blob synthétique sélectionné pour chaque couche. Un motif déjà utilisé par R650 est répété pour toutes les couches de géométrie GGML (18,20); les autres couches utilisent des blobs zéro aux tailles annoncées. Ce n'est pas un checkpoint et la répétabilité entre batches n'est pas un oracle indépendant.

## Commandes reproduites

- env VF_GGML_NATIVE_BUILD=<snapshot>/native-build cargo build --locked --offline --features native-ggml-cpu --bin vf-strata-native-bridge --bin vf-rank-session — PASS; un avertissement Rust non bloquant de champs inutilisés reste présent.
- c++ -std=c++17 -Wall -Wextra -Werror -pthread -I .runs/r651-build/adapters/strata_native_bridge .runs/r651-build/adapters/strata_native_bridge/pipe_client.cpp .runs/r651-build/adapters/strata_native_bridge/pool_hook.cpp research/R651/cpu-harness/native_48_pool_callback.cpp -o .runs/r651-build/native_48_pool_callback — PASS.
- python3 -B research/R651/cpu-harness/test_native_48_pool_callback.py avec les quatre chemins VF_R651_* décrits dans le harnais — PASS, R651 v3.

La validation utilise le snapshot R643 isolé et les sources pipe_client.* et pool_hook.* relues depuis la tête R650 3c39dd317c802802ce5614e7ac790fa6e0294bbb; leurs SHA-256 sont listés dans RESULT.json. Le build GGML provient du fournisseur CPU llama.cpp épinglé par R643 (3cf03257f219afbe7334045ff7c6a06ac68c627d, MIT). La documentation d'adaptation dans adapters/native_ggml/README.md référence Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf et indique MIT pour les deux projets; R651 ne copie ni ne compile les sources donneuses de Strata.

## Échecs conservés et reçus

Deux premières exécutions avec des identités différentes restent dans le dossier de reçus et ne sont jamais rejouées :

- La v1 a terminé huit callbacks (couches 0 à 7), puis le test a arrêté à tort la progression parce qu'il ne s'attendait à une sortie non nulle que pour la couche 0. Le générateur avait réutilisé le motif de couche 0 pour chaque couche ayant la même géométrie. Huit reçus sont conservés.
- La v2 a passé les 96 callbacks et l'arrêt C++ des sessions, mais le harnais a ensuite demandé au lanceur Python de renvoyer un second arrêt aux coordinateurs déjà arrêtés. Les 96 reçus sont conservés.
- La v3 utilise le chemin de fermeture du vrai launch() après que le callback a pris en charge l'arrêt. Elle a passé les 96 callbacks, le refus terminal, les deux niveaux de fermeture et la relecture des 96 identités.

Les trois lots portent des RUN_GUARD distincts et aucun ne peut être relancé par le harnais. Tous les reçus ont l'état RESERVED_UNKNOWN_NO_REPLAY; ils sont locaux à ce test, pas des réservations R53 partagées ou de production. Les deux erreurs de harnais ne sont pas masquées ni présentées comme des échecs du moteur.

## Portée et prochaine frontière

R651 qualifie la flotte C++ VoltaFabric raccordée à 48 vrais coordinateurs et sessions Rust synthétiques sur CPU. Il ne lance pas le binaire Strata donneur ni son callback de génération; aucun vrai checkpoint, modèle entier, GPU, RDMA, bail R53 de production ou worker inter-PC n'a été utilisé. Aucun runtime de vLLM, SGLang, TensorRT-LLM, Maya ou DeepInfer n'est ajouté ou qualifié ici. La réutilisation réellement intégrée reste le résident CPU GGML de llama.cpp épinglé en R643; R651 vérifie la frontière native et la gestion des sessions de VoltaFabric.

Pour connecter le véritable callback, il reste à obtenir et compiler les sources donneuses exactes de Strata et à lier leurs plans de couche/checkpoint aux identités des 48 sessions. Cela demande une nouvelle fixture admise; aucun modèle complet ne doit être chargé par ce test.
