# R716 — du gate DeepSeek à l'exécution MXFP4 groupée

## Changement

`DeepSeekV41RouteBatch` expose maintenant trois opérations qui gardent sa route et sa tentative privées et liées : préflight des owners MXFP4, regroupement des experts sélectionnés selon le dispatch rank/local, puis exécution groupée dans les Sessions chargées. L'appel d'exécution consomme le reçu de gate avec les reçus de LOAD; le caller ne recompose pas le `RoutingProblem` ou l'identifiant d'attempt entre ces étapes.

Les trois opérations délèguent aux frontières R713–R715 déjà qualifiées. Le préflight exige les six experts et leurs empreintes avant les Sessions; le regroupement refuse une modification de dispatch ou un budget insuffisant; l'exécution valide toutes les Sessions avant le premier EXEC. Le Collector R9 FP32 prépondéré applique le coefficient déjà calculé par le gate une seule fois, et la sortie reste atomique.

## Preuve CPU

Le nouveau test construit une fixture Safetensors de gate synthétique au format du modèle épinglé, fait produire au gate la route top-6 texte, la lie à un stamp complet et au plan de rangs deux-plus-quatre, construit six owners MXFP4 synthétiques, puis appelle uniquement l'API de `DeepSeekV41RouteBatch` pour préflight, regroupement et exécution. Cinq Sessions CPU chargent les artifacts; les six slots sont terminés. La sortie du Collector est comparée à `Mxfp4Expert::forward_reference`; le test impose un écart absolu maximal de `5e-4`.

- le test gate→grouped Session est passé en debug (1/1) et release (1/1);
- Clippy `-D warnings`, format Rust et `git diff --check` passent.

Ces résultats qualifient une couche de routage/gate sur CPU synthétique, pas la génération complète d'un modèle.

## Réemploi et limites

Réemploi interne effectif : `DeepSeekV41Gate`, le reçu de route, la lecture Safetensors MXFP4, les artifacts, le dispatch, le protocole Session, et le Collector R9 existants. La référence DeepSeek V4.1 Flash est épinglée à `314113f8a7dadb388215396ba5fbee2c610a04d6` et sa provenance MIT est conservée; aucun code tiers n'est copié dans cette adaptation.

La fixture de gate et les six poids d'experts sont synthétiques; aucun tensor gate ou expert de checkpoint réel n'a été chargé. Le plan deux-plus-quatre est synthétique, et n'est pas une preuve du registre réellement retenu. La preuve ne couvre pas attention GDN, cache KV, shared expert, tokenizer, multi-token generation, GPU/V100, Spark/RTX 5080, inter-PC/RDMA, R53 de production ni performance. Les formats MXFP4 K32, QPN2/NVFP4 K16 et IQ3 restent séparés. La CI GitLab ne remplace pas les reçus des tests Rust locaux.

## Suite

R717 : confronter le chemin gate→grouped-exec aux fichiers de checkpoint et au registre réellement conservés, en lecture strictement bornée; les preuves R703 montrent que les payloads top-6 379–382 manquent dans l'inventaire suivi. Tant qu'ils manquent, maintenir le refus avant Sessions/EXEC. Pas de téléchargement global, GPU, réseau de production, rejeu UNKNOWN, nettoyage ou relance automatique.
