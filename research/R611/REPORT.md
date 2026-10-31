# R611 — divergence dès la couche 0

Deux transactions CPU locales distinctes ont traité la première couche du checkpoint SC117 observé, chacune avec le premier token 248045 du prompt R609. Les deux utilisent exactement le même embedding vérifié par SHA-256, les mêmes poids observés et les mêmes admissions d'experts ; seuls les backends natifs scalaire et vectoriel diffèrent. Les deux workers de chaque transaction se sont fermés après exécution et le segment a validé une position.

Sur 10 240 valeurs f32 du résiduel de sortie de couche 0, 20 diffèrent, avec un écart absolu maximal de 9,6894800663e-6. C'est une divergence observée dès la première couche pour cette entrée, ce qui resserre l'analyse R610 ; le contrôle ne montre pas encore quelle projection, quantification ou contribution d'expert l'a provoquée et ne prouve pas quel chemin est correct. La comparaison NumPy vérifie les valeurs et leurs hashes, pas l'arithmétique indépendante du modèle. Les valeurs dérivées du modèle restent locales.

Chaque transaction utilisait un plan externe, un run et un registre neufs ; chacun des deux claims est vérifié et conservé dans `RESULT.json` avec le statut `RESERVED_UNKNOWN_NO_REPLAY`. Chaque scope était limité à 2 Gio, zéro swap et un cœur. Aucun GPU, worker inter-PC, registre R53 ou lease partagée n'a été utilisé. Les formats GGML propres au checkpoint ne sont pas assimilés à MXFP4 ou NVFP4.

Le nouveau diagnostic compile sur PC1 et PC6 ; la vérification NumPy, rustfmt et Clippy ciblés passent. Étape suivante : capturer les sorties individuelles des experts et les étapes intermédiaires de cette couche sur la même activation pour isoler la première divergence, puis évaluer la politique numérique avant distribution.
