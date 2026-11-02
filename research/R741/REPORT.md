# R741 — oracle d’IDs de tokens du prompt DeepSeek V4.1

Date : 2026-11-02

## Résultat

L’adaptateur R740 produit exactement les mêmes IDs que l’oracle CPU indépendant dans cinq cas : texte ordinaire; texte utilisateur contenant des chaînes ressemblant à des tokens de contrôle; puis modes de raisonnement `low`, `high` et `max`. Le test charge les octets du tokenizer V4.1 déjà épinglé par R740, puis vérifie aussi la réservation de contexte et la somme de contrôle retournée par l’adaptateur.

L’oracle n’appelle ni le renderer DeepSeek Recipe ni `PromptTokenizer` : il assemble les tokens de contrôle dans leur ordre explicite, retire le vocabulaire ajouté avant d’encoder le texte utilisateur et le texte d’effort, puis compare la séquence complète d’IDs. La référence est construite avec le tokenizer officiel épinglé et le contrat de prompt documenté par DeepSeek. Elle ne constitue pas une comparaison avec des logits ni une génération du modèle.

## Réutilisation et traçabilité

- Code testé : `src/deepseek_v41_prompt.rs`, feature Cargo `deepseek-v41-recipe`.
- Dépendances réutilisées par R740 : `deepseek-recipe-core` 0.2.0 et `deepseek-recipe-encoding` 0.2.0 (MIT), `tokenizers` 0.23.2 (Apache-2.0). R741 n’ajoute aucune dépendance.
- Le test couvre aussi la correction 0.2.0 des chaînes de tokens spéciaux fournies en texte ordinaire; l’issue amont documente le défaut antérieur et sa correction.
- Les octets tokenizer utilisés restent un fixture local temporaire, non commité à GitLab. Aucun poids ni checkpoint n’a été chargé.

## Vérification

Commande exécutée depuis `.runs/r740-full` :

```sh
CARGO_TARGET_DIR="$PWD/target" VF_DEEPSEEK_V41_TOKENIZER_JSON="$PWD/pinned-tokenizer.json" cargo test --locked --manifest-path ../r741-oracle/Cargo.toml --lib --features deepseek-v41-recipe reference_tests::one_turn_token_ids_match_independent_pinned_reference -- --ignored
```

Résultat : 1 test réussi, 0 échec. Deux avertissements `dead_code` proviennent des helpers de test SHA-256 dans le petit harness isolé. Le test exécute bien `DeepSeekV41PromptEncoder` et la bibliothèque de tokenizer, mais le crate shim n’est pas la compilation complète du workspace VoltaFabric. Aucun build complet n’est revendiqué.

## Limites

R741 ne qualifie que l’encodage d’un unique tour utilisateur texte. Historique, outils, JSON/schema, multimodal, décodage de réponse, règles d’arrêt, échantillonnage, logits, checkpoint complet, GPU, réseau inter-PC et performances restent hors preuve. Le test est ignoré par défaut car il requiert l’asset tokenizer épinglé; la commande ci-dessus l’active explicitement.

## Suite proposée

R742 : définir et tester le décodage borné des IDs de sortie ainsi que la distinction des tokens de fin V4.1 contre le tokenizer de référence, puis raccorder cette frontière au sampler/générateur existant. Ne pas annoncer une génération complète avant d’avoir relié ce chemin à une exécution de modèle compatible.

## Références amont

- [DeepSeek Recipe — dépôt et licence MIT](https://github.com/deepseek-ai/deepseek-recipe)
- [Documentation officielle du tokenizer V4](https://github.com/deepseek-ai/deepseek-recipe/blob/main/docs/tokenizer.md)
- [Issue amont sur les chaînes de tokens fournies comme texte](https://github.com/deepseek-ai/deepseek-recipe/issues/5)
- [Commit DeepSeek Recipe 0.2.0](https://github.com/deepseek-ai/deepseek-recipe/commit/93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170)
