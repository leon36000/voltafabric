# R744 — liaison CPU entre logits greedy, identité DeepSeek V4.1 et frontière EOS

## Résultat

R744 ajoute un sélecteur de prochain token greedy déterministe sur CPU et le relie au prompt encoder R742. Le nouvel objet refuse les logits qui ne correspondent pas au vocabulaire/tokenizer admis, à l’identité complète `run/model/plan/layer/batch/epoch` et à l’`attempt` courant. Il borne le nombre de tokens à la réservation du prompt, traite EOS comme terminal et n’autorise le passage à la frontière de streaming qu’après terminaison. Le préfixe transmis au stream exclut EOS conformément au contrat déjà testé en R742.

La règle greedy conserve le premier token au maximum en parcourant les ID dans l’ordre et en ne remplaçant le choix courant que pour un logit strictement supérieur. Un logit `-∞` est un candidat masqué admissible; NaN, `+∞` et une ligne sans candidat fini sont refusés sans ajouter de token. Une égalité stable ne signifie pas que la génération complète est déterministe: le backend de logits, les kernels, la précision et le reste du moteur ne sont pas couverts ici.

L’objet `EncodedPrompt` garde ses champs de lecture mais porte aussi une admission privée liant digest tokenizer, digest des IDs, limite de contexte et réservation. Le factory du sampler vérifie ce sceau avant de calculer; modifier ou fabriquer les champs publics après encodage entraîne un refus. Les IDs sont hashés comme `u32` little-endian, et la longueur est bornée avant ce contrôle.

## Réemploi amont

- Source examinée: `ggml-org/llama.cpp`, commit [`29c720eaf72fcf8e4e61a56a88b994d92668261d`](https://github.com/ggml-org/llama.cpp/tree/29c720eaf72fcf8e4e61a56a88b994d92668261d).
- Licence effective à ce commit: MIT ([LICENSE épinglée](https://github.com/ggml-org/llama.cpp/blob/29c720eaf72fcf8e4e61a56a88b994d92668261d/LICENSE)).
- Idée adaptée: parcours ordonné et remplacement strict `>` pour que le premier candidat gagne les égalités, visible dans [`llama-sampler.cpp`](https://github.com/ggml-org/llama.cpp/blob/29c720eaf72fcf8e4e61a56a88b994d92668261d/src/llama-sampler.cpp#L1054-L1061).
- Aucun code source llama.cpp n’est copié. L’adaptateur ajoute les contrôles propres à VoltaFabric (digest tokenizer épinglé, EOS/checkpoint, identité et tentative, réservation et refus atomique d’une ligne invalide).
- L’encodeur/tokenizer et le préfixe EOS-safe sont ceux de R742; R744 ne remplace pas le parseur DeepSeek Recipe.

## Vérification effectuée

Les véritables modules prompt et sampler ont été compilés dans un harness Rust isolé avec le feature stream actif, les dépendances exactes `deepseek-recipe` 0.2.0, `deepseek-recipe-core`/`deepseek-recipe-encoding` 0.2.0, `tokenizers` 0.23.2 et le SHA-256 VoltaFabric. Cargo fonctionne en mode offline; 9 tests passent et les trois tests utilisant le tokenizer épinglé restent ignorés. Cela compile aussi l’adaptateur vers le parser DeepSeek Recipe, sans prouver le contenu streaming avec vrai tokenizer. Les tests incluent les six contrats sampler, les vecteurs NIST SHA-256 et l’ordre du digest d’IDs. Ce contrôle a révélé puis corrigé une garde cfg préexistante qui masquait le helper `plain_ids` au test EOS sans feature stream. `rustfmt --check` passe sur le sampler, le prompt et le `lib.rs` isolé.

Les trois tests d’intégration ignorés vérifient avec le tokenizer épinglé le prompt oracle, l’admission, l’EOS, le refus d’une mutation d’IDs et le parser de stream. Ils restent ignorés tant que `VF_DEEPSEEK_V41_TOKENIZER_JSON` n’est pas fourni. Le harness compile prompt/sampler et feature stream réels, mais pas le crate complet. Le pipeline source GitLab #2626 est verte sur `5565a8b594745af4205728a76aa005c05534e400` (Gitleaks, Semgrep, Trivy et fallback). Ce lot ne charge aucun tokenizer ni checkpoint réel.

## Ce qui reste hors preuve

Ce sélecteur reçoit une ligne de logits déjà calculée. Il ne produit pas ces logits: aucun embedding, bloc Transformer, routage MoE, LM head, lecture de tenseurs de checkpoint, KV cache, streaming de génération complète, GPU, transport inter-PC ni comparaison de performance n’est qualifié par R744. La chaîne avec propriétaires réels demeure à raccorder au vrai fournisseur de logits; les résultats du harness CPU ne simulent pas ce fournisseur.

Aucun poids, route, coefficient, registre de propriétaires ou sortie de génération complète n’a été modifié. Aucune donnée GPU, inter-PC ou R53 n’a été utilisée.
