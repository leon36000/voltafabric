# R131 — en-tête complet d'un expert Qwen MoE

Une seule lecture HTTP Range strictement limitée à l'en-tête safetensors complet (octets 8 à 6 937 607) du premier shard de `nvidia/Qwen3.6-35B-A3B-NVFP4` à la révision `1355db6a052410cfd62085d94b58866fd0f2c3c5` a été effectuée. Code 206, intervalle et taille du shard ont été vérifiés avant lecture ; 6 937 600 octets d'en-tête reçus, SHA-256 `0489a7e3ccdbfd6fc395a83982c74f7ce0b4ce582b71bf73487114412843963f`. Les clés JSON dupliquées et les offsets hors shard sont refusés. Aucun octet de charge de tenseur n'a été lu, aucun modèle ou GPU n'a été lancé. Le SHA LFS du shard entier n'a pas été vérifié.

Pour la couche 0, expert 0, cet en-tête indique :

| Projection | `weight` U8 | `weight_scale` F8_E4M3 | Autres champs |
|---|---|---|---|
| gate | `[512, 1024]` | `[512, 128]` | `input_scale` et `weight_scale_2`, scalaires F32 |
| up | `[512, 1024]` | `[512, 128]` | mêmes types de scalaires |
| down | `[2048, 256]` | `[2048, 32]` | mêmes types de scalaires |

Les offsets et types déclarés exacts des douze champs sont dans `stdout.json`. Les trois codes occupent au total 1 572 864 octets, les trois échelles par bloc 196 608 octets et les six scalaires 24 octets, selon ces offsets. Cela n'établit ni les valeurs, ni leur validité, ni la correspondance numérique complète à l'adaptateur. Une future lecture partielle doit pré-déclarer les plages et sa limite, préserver les SHA de chaque portion et ne pas prétendre vérifier le SHA LFS du fichier entier ; un oracle indépendant et les contrats de coefficient restent nécessaires.

La résolution complète des dépendances 1Cat reste ouverte. Ce résultat ne constitue ni une couche exécutée, ni une qualification GPU ou un modèle complet. Voir [le dépôt épinglé](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/tree/1355db6a052410cfd62085d94b58866fd0f2c3c5).
