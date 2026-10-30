# R129 — candidats de checkpoint MoE, métadonnées uniquement

Le fichier MODELS.md du donneur Skinny épinglé nomme notamment deux checkpoints MoE NVFP4. Leur métadonnée publique Hugging Face et `config.json` ont été lus à une révision explicite ; aucun tenseur ni poids n'a été téléchargé. Les valeurs ci-dessous sont des déclarations des dépôts et ne prouvent pas leur chargement ni leur compatibilité effective avec VoltaFabric.

| Dépôt | Révision | Tailles des trois fichiers safetensors | Licence indiquée par les tags |
|---|---|---:|---|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | `1355db6a052410cfd62085d94b58866fd0f2c3c5` | 23 424 338 320 octets | Apache-2.0 |
| `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | `94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa` | 23 420 419 700 octets | MIT |

Les deux configurations indiquent l'architecture `Qwen3_5MoeForConditionalGeneration`, 40 couches, dimension 2048, 256 experts, 8 experts sélectionnés par token et dimension intermédiaire MoE 512. Les configurations annoncent `modelopt` et des groupes de poids float 4 bits de taille 16, avec aussi des groupes float 8 bits ; ce n'est pas une vérification des tenseurs ni de tous les champs de quantification. Les empreintes LFS déclarées des trois fichiers et les SHA des deux configs sont dans `stdout.json` et `configs.json`. Vérifier les licences effectivement applicables, la disponibilité et les octets avant utilisation ; un tag n'est pas une analyse juridique complète.

Ces deux candidats répondent mieux au périmètre MoE que le modèle dense par défaut du bootstrap. La taille de 23,4 GB concerne uniquement les poids déclarés : elle ne comprend ni tokenizer, buffers, cache KV, environnement ou pic de chargement et ne démontre pas la capacité d'un V100 ou d'une VM. Ne pas télécharger automatiquement les deux modèles. Avant d'en choisir un, rapprocher son architecture et ses fichiers de la couche prise en charge par le fork, vérifier les ressources de la cible et résoudre le runtime ; une seule couche puis la génération exigent leurs propres preuves.

Sources : [Qwen](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Ornith](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4), [MODELS.md du donneur épinglé](https://github.com/gilby/volta-nvfp4/blob/b1a2181702e609dc92591343e46c63d1324ade43/MODELS.md). Aucun modèle, GPU ou benchmark exécuté.
