# R118 — réutiliser un moteur complet autour de VoltaFabric

La comparaison de vitesse est mise en attente à la demande du propriétaire. Le prochain objectif produit est une génération complète qui utilise effectivement un fournisseur d'experts VoltaFabric.

## Décision d'ingénierie

La voie principale proposée est le raccord **vLLM / 1Cat-vLLM / Skinny déjà présent**. Le moteur donneur conserve le chargement du modèle, le tokenizer, l'attention, le cache KV, les opérations hors experts, le sampling et la boucle de génération. VoltaFabric conserve ses identités de calcul, la propriété des poids résidents, le placement des experts, les limites, les reçus et la collecte des résultats.

Cette décision est un choix de l'intégrateur fondé sur le code disponible. Elle ne constitue pas encore une intégration exécutée, ni une estimation du temps économisé. L'obtention de maturité dépend de l'utilisation effective du moteur donneur et de la qualification de la nouvelle frontière ; copier ses fichiers ne transmet pas automatiquement ses garanties.

```text
prompt → moteur donneur : tokenizer → embedding → attention / KV → routeur
                                                          ↓
                       frontière expert explicite : couche, token, slot,
                       expert, activation, coefficient, modèle et epoch
                                                          ↓
                      VoltaFabric : propriétaire résident / exécution
                                  → reçus → collecte ordonnée
                                                          ↓
                  moteur donneur : couches restantes → logits → sampling
```

## Pourquoi cette voie en premier

`adapters/skinny_moe.py` est déjà dérivé du donneur `gilby/volta-nvfp4@b1a2181702e609dc92591343e46c63d1324ade43`. Notre code expose `prepack_experts`, `moe_apply_local`, `moe_apply_dispatch_local`, `bind_rank_dispatch` et le propriétaire persistant de `adapters/persistent_gpu_owner.py`. Ces composants ne sont pas un nouveau tokenizer ou une nouvelle boucle d'attention.

Le fichier donneur `fork_patches/modelopt.py`, téléchargé au commit exact, importe `vllm.model_executor.layers.quantization.skinny_moe`. Ses deux appels sont `prepack_experts(layer)` et `moe_apply(layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input)`. L'analyse AST a confirmé que les signatures présentes ici acceptent ces arités. Le fichier donneur `marlin.py` expose aussi `_get_skinny_ext` et `_qpn_prepack`. Cette preuve est statique : elle ne remplace pas l'import du runtime installé, l'export natif `gemm_qpn2_moe`, ni un calcul réel de couche.

Le donneur épinglé référence 1Cat-vLLM **1.2.2**, Python **3.12** et CUDA **12.8**. Le vLLM officiel actuel annonce une capacité CUDA minimale **7.5** ; V100 est SM70. La première voie V100 doit donc conserver un fork compatible et une version précise. Installer le dernier vLLM ne constituerait pas une mise à jour équivalente.

## Les frontières à garder distinctes

| Chemin | Format et emplacement | Conséquence d'intégration |
|---|---|---|
| Skinny existant | NVFP4 bloc16, échelles E4M3, plans QPN2 ; tenseurs PyTorch GPU | Utiliser un checkpoint compatible et le contrat de propriété/dispatch déjà présent. |
| Worker R13/R76/R80 | MXFP4 bloc32 ; ABI FP32 hôte, admission BF16 exacte ; coefficients appliqués dans l'expert | Il ne remplace pas directement une fonction PyTorch recevant des tenseurs GPU. Transfert, stream, durée de vie et point de pondération doivent être explicites. |
| Pack SC117 Strata existant | IQ3_S / autres formats natifs ; ses activations et kernels propres | R110 exclut la conversion universelle exacte vers un seul couple E2M1/E8M0. Le pack ne doit pas être présenté comme MXFP4. |

L'appel pratique `moe_apply` possède un combine PyTorch `index_add_`. Il ne faut pas en déduire l'ordre strict du Collector Rust sur GPU. Pour une exécution distribuée VoltaFabric, la tranche doit utiliser les sorties par slot de `moe_apply_dispatch_local` ou du propriétaire persistant et la collecte ordonnée existante. La démonstration d'une génération locale qui sélectionne le hook Skinny et la démonstration de l'intégration du coordinateur Rust sont deux étapes explicites.

Le donneur ne passe pas notre masque d'ownership supplémentaire. Un premier périmètre local ne peut donc pas prétendre qualifier l'EP partitionné ; celui-ci exige un `RankDispatch` courant, son masque exact et les identités complètes avant l'appel. La compaction dynamique actuelle n'est pas qualifiée pour les graphes CUDA : commencer par un chemin d'exécution explicitement admis, sans reprendre des gains de graphes amont.

## Autres donneurs retenus

| Projet | Ce qu'il pourrait fournir | Position actuelle |
|---|---|---|
| **Strata** | Modèle complet, état KV/GDN, génération ; callbacks experts `PoolFn` / `PoolMultiFn` | Alternative concrète si l'objectif prioritaire devient le modèle SC117 IQ3 déjà présent. Il faudrait conserver son backend IQ3 sous un type de fournisseur explicite, ou choisir des poids compatibles VF. |
| **llama.cpp / GGML** | Tokenizer, formats GGUF, gestion de contexte, sampling, backends C/C++ | Intéressant pour une cible portable. Un backend GGML et son ordonnanceur ont un contrat buffers/événements/opérations ; remplacer une seule multiplication ne remplace pas notre chaîne gate/up/activation/down. |
| **vLLM officiel** | Plugins de modèles hors arbre, cache KV, batching et service | Voie ultérieure pour les GPU et modèles compatibles avec sa version choisie. `CustomOp.forward_oot` n'active pas automatiquement un opérateur sur la plateforme CUDA standard. |
| **SGLang** | Runtime et frontière MoE `FusedMoE → quant_method.apply → MoeRunnerCore` | Candidat ultérieur à un fournisseur tensoriel. L'interface interne et les backends admis doivent être épinglés ; aucune qualification SM70 n'est déduite ici. |

### Obstacles précis dans Strata

Les headers locaux `session.hpp` et `layer.hpp` sont identiques entre les révisions `4b9e05a2a806a3fe24f7b2560d98626c37dade35` et `296b59d2f73c8f04a805df29c2fa97655adace78` déclarée par l'installation R107. Cela relie les interfaces source ; cela n'établit pas la correspondance du binaire compilé.

* `PoolFn` reçoit des activations FP32 hôte et produit des lignes `[k,hidden]`. Le pack natif IQ réel emprunte `PoolMultiFn`, avec `layer` et des lignes `[n_tok*k,hidden]` ; le chemin single-token refuse les native IQ packs dans la source étudiée.
* Le chemin single-token ignore `weights` dans son pool et les applique lors de `moe_combine_parts`. Le chemin multi a son propre placement de coefficient. Il faut qualifier chaque chemin avant de raccorder un fournisseur VF déjà pondéré ; aucun coefficient ne peut être appliqué deux fois.
* Les activations FP32 du modèle ne sont pas garanties BF16 exactes. Les arrondir silencieusement pour satisfaire l'admission R76 modifierait le calcul.
* Les callbacks sont `void`. Dans le chemin `session_loop` étudié, la copie vers `y_miss` et le graphe suivant peuvent suivre le callback avant contrôle du latch d'erreur du pool. Un adaptateur doit porter une erreur consommée avant copie/combinaison/publication, pour préserver l'échec atomique VF.

## Tranches de livraison

1. **Admission du moteur donneur.** Identifier un environnement isolé Python3.12 réellement installé, sa version 1Cat, les fichiers exacts et exports natifs. Vérifier le checkpoint compatible à partir des chemins connus ; ne pas lancer le script de bootstrap qui remplace des fichiers dans une installation active.
2. **Une vraie couche.** Faire passer une couche et des routes gelées par le hook exact, démontrer que le fournisseur VF est sélectionné, vérifier poids résidents, coefficients une fois, ordre et refus. Le nombre de symboles ou l'import du module ne sont pas une exécution.
3. **Une génération complète.** Le moteur donneur prend un prompt et produit des tokens avec ce hook réellement utilisé. Conserver les erreurs d'origine, les reçus et une trace de sélection ; aucune sortie partielle lors d'un résultat expert invalide.
4. **Coordinateur / placement.** Raccorder le dispatch courant et le Collector Rust aux sorties par slot ; fermer les erreurs et durées de vie avant une distribution inter-PC. Les preuves SSH antérieures ne qualifient pas RDMA.

Le premier modèle doit être nommé et ses octets/format admis avant calcul. Il n'est pas justifié de télécharger un second modèle de dizaines de Go avant de vérifier les assets et environnements déjà disponibles.

## Lecture fraîche PC1

L'accès SSH `pc1-lan` a réussi après la confirmation du propriétaire. La sonde a lu uniquement les métadonnées des distributions du Python par défaut et deux fichiers approuvés. Elle n'a importé aucune bibliothèque CUDA ou de modèle.

* Python par défaut : `/usr/bin/python3`, **3.14.4**. `torch`, `vllm`, `1cat-vllm`, `sglang` ne sont pas déclarés dans cet interpréteur ; cela ne prouve pas leur absence dans d'autres environnements.
* Binaire Strata existant : 75 892 384 octets, SHA256 `3f34217fdfecfd44ef25c2124e9388911bd50131776b7e8307c3d667a1f2632d`, lecture stable.
* `BUILD.json` adjacent : 271 octets, SHA256 `cf36bd5489fa91b94d33f0b9623fedbe33925f1052ce884f31a623b9bb01ba9e`, identique au relevé R107. Un hash du binaire établit ses octets, pas encore son lien au source déclaré.

Zéro octet de payload modèle lu, aucun moteur démarré, aucun GPU ou NVML interrogé. Les reçus sont sous `.runs/upstream-reuse-integration-r118-20261028` sur PC6.

## Licences et maintenance

llama.cpp et Strata déclarent MIT ; vLLM et SGLang déclarent Apache-2.0. La chaîne Skinny est mixte : patches vLLM sous Apache-2.0, noyaux propres MIT, helpers tiers avec leurs notices. Conserver chaque licence, copyright, attribution et avis de modification. La licence du modèle reste distincte de celle du moteur. La licence de publication du code propre VoltaFabric n'a pas encore été choisie dans le socle ; cela doit être résolu avant diffusion publique.

## Sources

* [Donneur Skinny épinglé](https://github.com/gilby/volta-nvfp4/tree/b1a2181702e609dc92591343e46c63d1324ade43), [modelopt.py](https://github.com/gilby/volta-nvfp4/blob/b1a2181702e609dc92591343e46c63d1324ade43/fork_patches/modelopt.py), [NOTICE](https://github.com/gilby/volta-nvfp4/blob/b1a2181702e609dc92591343e46c63d1324ade43/NOTICE).
* [vLLM plugins](https://docs.vllm.ai/en/latest/design/plugin_system/), [CustomOp](https://docs.vllm.ai/en/latest/design/custom_op/), [exigences GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).
* [Backend GGML](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-impl.h).
* [SGLang MoE runner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/base.py).
* [Strata session épinglée](https://github.com/Niko1221/Strata/blob/4b9e05a2a806a3fe24f7b2560d98626c37dade35/include/strata/core/session.hpp), [verify.hpp](https://github.com/Niko1221/Strata/blob/4b9e05a2a806a3fe24f7b2560d98626c37dade35/include/strata/core/verify.hpp).

Les liens `main`, `master`, `latest` et `stable` ont été consultés le28octobre2026 pour l'orientation. Le code effectivement choisi devra avoir un commit et des hashes fixes avant installation ou intégration.
