# R717 — planifier les plages MXFP4 depuis la route du gate

## Changement

`DeepSeekV41RouteBatch::plan_selected_mxfp4_payloads` utilise directement l'ordre des six experts et l'identité de tentative produits par le gate. Il lie ces données au planateur Safetensors épinglé, au stamp complet, aux coefficients et au dispatch rang/local via l'empreinte de route existante. Le plan retourne les six plages exactes de chaque expert, le shard et le volume agrégé, sous une limite explicite.

Le nouveau chemin ne ré-accepte pas une liste d'experts fournie séparément. Il vérifie l'index et l'en-tête épinglés, la forme et les noms des 36 tenseurs et les offsets, puis refuse un budget trop petit avant de retourner un plan utilisable. Cette méthode ne fait aucune I/O de payload et ne crée pas de Session; l'orchestration appelante doit la placer avant ces effets. Elle ne prouve pas que les octets de poids existent.

## Vérification CPU

Le test fait produire `[0, 383, 382, 381, 380, 379]` à la fixture gate synthétique, puis passe l'index complet suivi et l'en-tête réel suivi du shard `model-00003-of-00048.safetensors` au planificateur. Les 36 plages correspondent aux six tenseurs par expert et totalisent 112 803 840 octets dans un shard annoncé de 7 389 759 032 octets. Une limite inférieure d'un octet est refusée avec `ARTIFACT_BUDGET`.

- test ciblé debug : 1 réussi;
- test ciblé release : 1 réussi;
- Clippy `-D warnings`, `cargo fmt --check` et `git diff --check` : réussis.

Le gate reste synthétique. Aucun octet des payloads d'experts n'a été lu. Aucun worker/Session, GPU, réseau inter-PC, registre R53 de production ou modèle complet n'est utilisé.

## Réemploi, provenance et limites

Cette tranche compose le gate CPU VoltaFabric avec son planificateur sélectif Safetensors existant (`safetensors` 0.8.0, Apache-2.0) et les métadonnées suivies du modèle `deepseek-ai/DeepSeek-V4.1-Flash` à `314113f8a7dadb388215396ba5fbee2c610a04d6` (MIT). Aucun code donneur n'est copié. Les formats MXFP4 K32, QPN2/NVFP4 K16 et IQ3 restent distincts.

R703 et R709 établissent que les payloads d'experts 379–382 et le shard complet ne sont pas dans l'inventaire suivi. L'index et l'en-tête ne constituent pas une preuve de résidence ni d'intégrité des poids. L'admission de six owners réels reste donc fermée. Cette tranche ne qualifie ni la compatibilité matérielle V100 de ce modèle, ni les autres composants d'une génération complète.

## Suite

R718 : rapprocher les plans issus d'une route des reçus de disponibilité et des hashes de payload réellement conservés. Préflighter les six experts ensemble avant toute lecture coûteuse; refuser toute sélection incomplète. Aucun téléchargement global, GPU, production réseau, rejeu UNKNOWN, suppression ou relance automatique.
