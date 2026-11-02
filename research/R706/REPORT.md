# R706 — porter l’identité gate jusqu’à la Session MXFP4

## Changement livré

`DeepSeekV41RouteBatch::into_mxfp4_session_batch` construit le `Batch` déjà utilisé par la Session MXFP4 persistante. Il conserve le `RoutingProblem`, l’ordre des routes, les coefficients et l’`attempt` du gate sans rerouter ni renormaliser. Il accepte un seul token, l’entrée complète et déjà exactement BF16; il refuse les entrées partielles ou qui nécessiteraient un arrondi.

La Session MXFP4 exige désormais un `attempt` explicite et non nul par batch, refuse les doublons avant d’ouvrir les artefacts, transmet cette valeur à l’EXEC et la rend dans `BatchReport`. Elle ne fabrique plus l’identité d’essai depuis l’index du batch. Le Collector prépondéré FP32 et les validations d’identité existants restent le chemin d’exécution.

## Réutilisation et sources

- Le raccord réutilise le gate VoltaFabric adapté au commit DeepSeek-V4.1-Flash `314113f8a7dadb388215396ba5fbee2c610a04d6` et la Session MXFP4 persistante déjà présente. Le dépôt conserve une copie MIT du LICENSE du modèle. R706 ne copie aucun code externe.
- Le scan arXiv demandé a fait ressortir **Fiddler** (placement CPU/GPU selon le scénario), **MoE-Lightning** (pipeline CPU/GPU/I/O) et **DAOP** (pré-calcul CPU guidé par les activations); les articles donnent des hypothèses de planification, pas des mesures VoltaFabric. **JANUS** sépare les pools attention et experts à une échelle de service plus grande. Les chiffres publiés sont ceux des auteurs et leurs matériels ne valident ni V100 ni la liaison inter-PC ciblée.
- L’idée retenue cette tranche est d’exiger la route complète et l’identité avant le lancement persistant. Le préchargement prédictif et le calcul CPU spéculatif ne sont pas introduits. Les dépôts des méthodes de recherche ne sont pas importés: leur compatibilité et licences n’ont pas été établies pour une reprise de code.

## Vérification

- Test unitaire: unicité et valeur non nulle des attempts — **1 réussi**.
- Test ciblé `r702_deepseek_gate_projection` — **5 réussis**, dont l’adaptateur gate→Session, l’égalité bit à bit des coefficients et le refus d’entrée inexacte.
- `cargo clippy --offline --locked --lib -- -D warnings`, `rustfmt --check` sur les fichiers touchés et `git diff --check` — réussis.
- Le test a tourné dans le worktree disponible basé sur R699, qui contenait déjà les sources locales des vagues suivantes. Les fichiers gate, test et Session-series avant patch correspondent exactement à leurs objets GitLab R705; `src/session/mxfp4.rs` local diffère du fichier R705, mais R706 ne le modifie pas et sa signature d’EXEC a été relue sur GitLab. Le pipeline R706 #2440 est un pipeline de scans, pas une compilation Rust.

## Limites

Aucun worker Session n’a été lancé et aucun checkpoint n’a été ouvert. R703 indique que le gate réel et les quatre experts 379–382 manquent; ce raccord ne permet donc pas une génération DeepSeek complète. Il ne qualifie ni GPU, ni distribution inter-PC, ni transport de production. Le chemin reste refusé tant que gate, six profils/payloads résidents et admission mémoire cohérente ne sont pas présents.

## Traçabilité

Code publié au commit `158eecd91e2bb7d2c1f32de5e13ad43aabffe129` sur la branche privée `r706-deepseek-mxfp4-session-attempt-20261102`; les trois fichiers source/test ont été relus identiques depuis GitLab. Aucun changement à `main`, aucun retry de pipeline, aucune relance automatique.

### Papiers arXiv examinés

- [Fiddler, arXiv:2402.07033 v3](https://arxiv.org/abs/2402.07033v3)
- [MoE-Lightning, arXiv:2411.11217 v1](https://arxiv.org/abs/2411.11217v1)
- [DAOP, arXiv:2501.10375 v3](https://arxiv.org/abs/2501.10375v3)
- [JANUS, arXiv:2512.13525 v4](https://arxiv.org/abs/2512.13525v4)
