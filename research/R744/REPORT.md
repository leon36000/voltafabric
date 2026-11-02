# R744 — liaison CPU entre logits greedy, identité DeepSeek V4.1 et frontière EOS

## Résultat

R744 ajoute un sélecteur de prochain token greedy déterministe sur CPU et le relie au prompt encoder R742. Le nouvel objet refuse les logits qui ne correspondent pas au vocabulaire/tokenizer admis, à l’identité complète `run/model/plan/layer/batch/epoch` et à l’`attempt` courant. Il borne le nombre de tokens à la réservation du prompt, traite EOS comme terminal et n’autorise le passage à la frontière de streaming qu’après terminaison. Le préfixe transmis au stream exclut EOS conformément au contrat déjà testé en R742.

La règle greedy conserve le premier token au maximum en parcourant les ID dans l’ordre et en ne remplaçant le choix courant que pour un logit strictement supérieur. Un logit `-∞` est un candidat masqué admissible; NaN, `+∞` et une ligne sans candidat fini sont refusés sans ajouter de token. Une égalité stable ne signifie pas que la génération complète est déterministe: le backend de logits, les kernels, la précision et le reste du moteur ne sont pas couverts ici.

## Réemploi amont

- Source examinée: `ggml-org/llama.cpp`, commit [`29c720eaf72fcf8e4e61a56a88b994d92668261d`](https://github.com/ggml-org/llama.cpp/tree/29c720eaf72fcf8e4e61a56a88b994d92668261d).
- Licence effective à ce commit: MIT ([LICENSE épinglée](https://github.com/ggml-org/llama.cpp/blob/29c720eaf72fcf8e4e61a56a88b994d92668261d/LICENSE)).
- Idée adaptée: parcours ordonné et remplacement strict `>` pour que le premier candidat gagne les égalités, visible dans [`llama-sampler.cpp`](https://github.com/ggml-org/llama.cpp/blob/29c720eaf72fcf8e4e61a56a88b994d92668261d/src/llama-sampler.cpp#L1054-L1061).
- Aucun code source llama.cpp n’est copié. L’adaptateur ajoute les contrôles propres à VoltaFabric (digest tokenizer épinglé, EOS/checkpoint, identité et tentative, réservation et refus atomique d’une ligne invalide).
- L’encodeur/tokenizer et le préfixe EOS-safe sont ceux de R742; R744 ne remplace pas le parseur DeepSeek Recipe.

## Vérification effectuée

Les véritables modules `src/deepseek_v41_sampler.rs` et `src/deepseek_v41_prompt.rs` ont été compilés dans un harness Rust isolé avec les dépendances épinglées `deepseek-recipe-core`/`deepseek-recipe-encoding` 0.2.0 et `tokenizers` 0.23.2, en mode offline, plus un `Stamp` et le type d’erreur de même forme que VoltaFabric. Les six tests sampler passent: égalité, EOS terminal et non-rejouable, limite de longueur, identité/tentative/forme, valeurs non finies et construction invalide. Deux tests avec tokenizer externe restent ignorés. Ce contrôle a également révélé puis corrigé une garde cfg préexistante qui masquait le helper `plain_ids` au test EOS alors que le feature stream n’était pas activé. `rustfmt --check` passe sur le sampler, l’adaptateur prompt modifié et le `lib.rs` isolé.

Le test d’intégration ajouté au test ignoré de l’encodeur vérifie avec le tokenizer épinglé la création du sampler, un choix EOS et l’absence d’EOS dans le préfixe final. Il reste ignoré tant que `VF_DEEPSEEK_V41_TOKENIZER_JSON` n’est pas fourni. Le harness compile le vrai code prompt/sampler, mais pas le crate complet ni son feature stream; la pipeline GitLab reste une vérification distincte. Ce lot ne charge aucun tokenizer ni checkpoint réel.

## Ce qui reste hors preuve

Ce sélecteur reçoit une ligne de logits déjà calculée. Il ne produit pas ces logits: aucun embedding, bloc Transformer, routage MoE, LM head, lecture de tenseurs de checkpoint, KV cache, streaming de génération complète, GPU, transport inter-PC ni comparaison de performance n’est qualifié par R744. La chaîne avec propriétaires réels demeure à raccorder au vrai fournisseur de logits; les résultats du harness CPU ne simulent pas ce fournisseur.

Aucun poids, route, coefficient, registre de propriétaires ou sortie de génération complète n’a été modifié. Aucune donnée GPU, inter-PC ou R53 n’a été utilisée.
