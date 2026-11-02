# R742 — DeepSeek V4.1 output stop and stream boundary

## Résultat

Le modèle épinglé `DeepSeek-V4.1-Flash` (révision `314113f8a7dadb388215396ba5fbee2c610a04d6`) expose `eos_token_id = 1`. Le tokenizer épinglé nomme ce token `<｜end▁of▁sentence｜>`. Le code vérifie que la configuration et le tokenizer admis ont le même ID/EOS. `generated_prefix` enlève le premier EOS et tout suffixe impossible, puis expose un drapeau de fin à l'appelant. Quand le jeton apparaît, l'appelant doit arrêter l'échantillonneur et signaler `Stop`.

L'adaptateur peut maintenant créer `deepseek_recipe::stream::StreamProcessor` en lui donnant le tokenizer déjà admis. La bibliothèque officielle maintient la différence entre un token spécial réellement généré et une suite de caractères utilisateur identique. La couche VoltaFabric ne réimplémente donc pas le parseur de protocole du donneur.

## Provenance et licences

- `deepseek-recipe` 0.2.0, MIT : API officielle de parsing de flux, optionnelle et derrière `deepseek-v41-stream`.
- `deepseek-recipe-core` et `deepseek-recipe-encoding` 0.2.0, MIT; `tokenizers` 0.23.2, Apache-2.0 : dépendances de l'adaptateur de prompt existant.
- Aucun code source donneur n'est copié. Le fichier Cargo.lock consigne les versions résolues.

Sources primaires consultées :
- https://github.com/deepseek-ai/deepseek-recipe/blob/main/docs/tokenizer.md
- https://github.com/deepseek-ai/deepseek-recipe/blob/main/deepseek-recipe/Cargo.toml
- https://github.com/deepseek-ai/deepseek-recipe/blob/main/docs/streaming.md
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json

## Adaptation et vérification

Le parseur officiel ajoute une dépendance transitive `jsonschema -> fraction -> num-bigint >=0.4.8`, incompatible avec l'épingle VoltaFabric `num-bigint =0.4.6`. La résolution du manifeste a reproduit le conflit. R742 met à jour cette épingle exacte à `0.4.8`, conserve les versions verrouillées existantes et ajoute les 41 paquets requis par la fonction optionnelle; le lock passe de 212 à 253 paquets. Le seul changement de version dans les paquets déjà verrouillés est `num-bigint` 0.4.6 → 0.4.8.

Commande d'oracle CPU isolé (code source exact de R742, tokenizer épinglé externe, `StreamProcessor` officiel) :

```sh
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/.runs/r740-full/target" \
VF_DEEPSEEK_V41_TOKENIZER_JSON="$PWD/.runs/r740-full/pinned-tokenizer.json" \
CARGO_NET_OFFLINE=true cargo test --locked \
  --manifest-path .runs/r742-oracle/Cargo.toml --lib \
  --features deepseek-v41-stream reference_tests -- --ignored
```

Résultat : 3 tests réussis : EOS/config/tokenizer, contrôle du suffixe post-EOS, et parsing par le donneur où `</think>` spécial ouvre la réponse mais le même texte encodé comme tokens ordinaires reste du contenu. Le test R741 garde aussi ses cinq cas de parité d'encodage de prompt.

`cargo metadata --locked --no-deps` a accepté le manifeste et le verrou ajustés. `cargo test --lib` et `cargo check --lib` sur `.runs/r740-full` ne sont pas des constructions complètes : cette copie locale de vérification ne contient pas les 29 modules déclarés par `src/lib.rs`. Les dépendances du feature sont résolues et compilées avant l'échec sur ces fichiers absents. Ce signal n'est pas présenté comme une erreur du code R742 ni comme une réussite du workspace complet. Le pipeline GitLab sur la branche complète doit trancher la construction intégrée.

## Portée non qualifiée

Pas de poids du checkpoint chargé, de logits, d'échantillonneur, de génération, de transport HTTP/SSE en service, de GPU, de réseau inter-PC, de R53 production ou de performance. Le test d'output couvre l'adaptation optionnelle et son contrat de tokens, pas une génération complète ni toutes les sorties multimodales/outils/schémas.

## Suite

Vérifier le pipeline et la relecture GitLab R741, puis publier R742 sur une nouvelle branche sans réécrire l'historique. Faire exécuter le pipeline complet avant de revendiquer l'intégration du changement de verrou.

## Reçu GitLab

Le commit source `c6ac701e8bbe92767a6df7eba1f96c2ff4c0572b` a passé le pipeline GitLab #2610. Les quatre tâches (`gitleaks`, `semgrep`, `trivy`, `ci_fallback`) sont en succès. Les fichiers de la source et les sept entrées de `SHA256SUMS` avaient été relus exactement avant ce pipeline.

Prochaine tranche : adapter une idée arXiv de placement fondée sur les coûts ou de chevauchement dans le planner CPU existant, avec plans stables, poids résidents protégés par les capacités mémoire et routes inchangées.
