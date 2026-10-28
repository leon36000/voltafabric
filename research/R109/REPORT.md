# R109 — profil Aider/R106 privé, qualification CPU partielle

**Statut : `CPU_PARTIAL_PROFILE_NO_REAL_POST_NOT_READY`.** Le profil réutilise les octets exacts du relais R106, adapte le host/container Aider R99 et lit les nouveaux reçus `.terminal.json`. Il n'est pas autorisé à soumettre un modèle réel : la preuve de lancement du vrai Aider avec ce code et le lien externe du launcher restent ouverts.

Un plan unique et externe lie l'identité future `r109-terminal-audit-20261028-a`, l'alias déjà connu `cloud-deepseek-flash-1017`, max_tokens4096, les octets des trois sources, le texte de mission, le contrat public et le stub d'une tâche distincte. La tâche future, utile à l'exploitation, résume des diagnostics R106 en une structure limitée sans exposer de messages. Ses neuf tests indépendants sont préparés hors `mission/` avant toute requête ; le stub ne les passe pas. Le premier rouge CPU (exit1, neuf tests,31 erreurs/subtests dues à `NotImplementedError`) n'avait pas de stdout persistant ; un deuxième rouge documenté conserve la sortie complète. Aucun code de modèle n'a été obtenu ou exécuté.

Le lanceur `launch_snapshot.py` lit le plan borné et vérifie son SHA externe, lit/compile les mêmes octets des sources pinées et vérifie les hashes de la mission. Le host fait une seconde vérification du plan avant START exclusif et utilise des snapshots des sources exécutées. **Le lanceur lui-même n'est pas encore lié à un chargeur externe hashé/memfd ; une simple commande Python ne qualifie donc pas la chaîne entière.** Le staging futur devra consommer les octets du launcher vérifiés, lier aussi son hash au contrat de soumission, préparer un sibling PC1 exclusif et découvrir l'alias avant tout POST. Aucune route, clé ou configuration de gateway n'a été modifiée ici.

Le host exige maintenant un reçu R106 terminal valide, lié à la réservation par run_id, modèle, taille et SHA exacts. Une erreur de relais est un échec conservé ; reçu absent, altéré ou états contradictoires donnent UNKNOWN. Un reçu de calcul complet est marqué **PROCESS_RESULT_PENDING_CLEANUP_NO_REPLAY**, pas accepté. Après contrôle du cleanup seulement, une sortie modifiée et liée à son SHA est **EDIT_RECEIVED_INDEPENDENT_TESTS_PENDING** ; elle reste `quality_accepted=false` jusqu'aux tests indépendants isolés. Aucun succès de modèle n'est déduit d'un code Docker seul.

## Revue et corrections réellement exercées

- La lecture de mission, contexte et fichiers de travail utilise `O_NOFOLLOW`, `O_NONBLOCK`, `fstat` fichier régulier, plafond par fichier et borne pendant l'itération. Tests FIFO réel, symlinks, fichier trop grand, nombre de fichiers et doublon de cible.
- Un Docker lancé sans cidfile reste `container_identity_unknown_no_cidfile` : aucune absence n'est fabriquée et aucun conteneur n'est supprimé par nom. Avec cidfile, seule l'identité hexadécimale propre est interrogée/supprimée. Un échec de cleanup reste UNKNOWN.
- La réponse reçue, la source candidate réellement lue, l'état du processus et le cleanup sont liés. Un fichier inchangé, un hash faux ou un fichier supplémentaire ne deviennent pas un correctif ; les tests externes restent requis.
- Les entiers de tailles, statut HTTP et liste `proxy_statuses` sont strictement typés, même si Python considère `200.0 == 200`. Les captures pipes du host et d'Aider sont bornées pendant lecture, avec délai et reap.

Les trois versions du host/plan sont conservées sous `before_review/`; les corrections ont été motivées par une revue indépendante. Le premier lot CPU du profil faisait 15 tests ; le lot suivant26 ; le final **28 tests exit0** dans `cpu-tests-r3.stderr`. Un test supplémentaire, `test_relay_roundtrip`, utilise un vrai processus client, un vrai socket Unix, le vrai relais R106 et un faux serveur HTTP local : réponses `[200,409]`, un seul forward, threads/processus arrêtés. L'état Aider de ce test est **simulé** ; il ne qualifie pas Aider réel ni un modèle.

## PC1 : tentative de qualification CPU interrompue avant script distant

`stage_cpu_pc1.py` a été lancé une fois avec les sources initiales. `PC1_CPU_TERMINAL.json` conserve transport SSH **exit255**, stderr « Connection timed out during banner exchange ». L'authentification et le script distant n'ont pas été atteints ; aucune preuve que le vrai Aider ait démarré, aucun POST modèle et aucune clé utilisée. Aucun nouveau SSH ni rejeu n'a été effectué. `PC1_CPU_START.json` conserve le SHA du code initial et du plan antérieur ; il ne qualifie pas les sources corrigées. Le résultat initial et `before_review/` ne sont pas réécrits.

## Restant avant une soumission

1. Épingler le SHA du launcher par un chargeur externe qui exécute les octets vérifiés, puis tester sa mutation et le lien plan/source/mission sur la vraie machine d'exécution.
2. Vérifier disponibilité/ressources et alias sur PC1 lorsque l'accès revient, sans rejouer R99. Réaliser une preuve true-Aider + faux upstream de ce **candidat final**, avec sortie bornée et nettoyage prouvé ; aucun résultat actuel ne remplace cette porte.
3. Faire relire la chaîne complète et préparer un staging exclusif avec reçu d'opération ; une issue incertaine est conservée sans retry.
4. Soumettre au plus une nouvelle identité explicitement autorisée pour la tâche R109, puis évaluer son candidat isolément sans accès réseau, clé ou écriture aux données partagées. Les tests déjà préparés devront réussir avant toute promotion.

Le scope est CPU privé ; aucun fichier moteur canonique, GPU, modèle, API distante ou service de voisin n'a été modifié ou exécuté. Le statut R99 reste `AIDER_FAILED_NO_REPLAY` ; son diagnostic causal demeure INCONCLUSIVE.
