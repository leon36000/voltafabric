# R678 — Invocation Qwen2MoE par couche

Donneur examiné : `ggml-org/llama.cpp`, commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, licence MIT. La couture exacte reste celle de R672/R677 : Qwen2MoE appelle `build_moe_ffn`; ce dernier possède à la fois l'activation d'origine, les IDs dynamiques top-k, les poids post-gating définitifs et les sorties brutes des experts avant multiplication des coefficients. La branche expert shared et le résiduel sont construits séparément dans le graphe Qwen2MoE.

R678 ajoute `Qwen2MoERouteInvocation`, adaptateur CPU C++ autour des primitives déjà testées `RouteGraphTransaction` et `RouteGraphInvocation`. Il prend une identité complète de batch et une projection `expert_id -> owner_kind`, crée une transaction fraîche pour chaque couche, transmet les tenseurs dynamiques sans réordonner les slots et conserve les transactions jusqu'à la validation hôte. Une invocation avec des couches manquantes, dupliquées ou hors plage ne peut pas publier son résultat. Les poids finaux restent inchangés et le Collecteur les applique une fois.

Le CTest nouveau construit un graphe GGML synthétique à deux couches avec IDs et propriétaires différents. Il vérifie la capture par couche, l'ordre des slots, les coefficients exacts, la sortie scalaire indépendante `[17.4, 19.4]`, et le refus hôte quand une des deux couches attendues manque. Avec les tests de transaction, capture dynamique et VFR3 existants, les quatre CTests ciblés passent sur CPU AArch64, C++17 et avertissements traités en erreurs.

## Portée

Cette tranche prépare et vérifie le contrat d'invocation destiné au callback Qwen2MoE. Elle ne modifie pas le source du donneur et ne prétend pas que le callback est déjà branché dans `llama_context::process_ubatch` ou `llm_graph_context::build_moe_ffn`. Pas de checkpoint réel, génération complète, GPU/graphe CUDA, réseau inter-PC/RDMA, registre R53 de production ou comparaison de performance.

Prochaine étape R679 : appliquer un patch étroit au llama.cpp épinglé qui active l'adaptateur uniquement pour Qwen2MoE, crée l'identité et les transactions pour chaque `process_ubatch`, désactive le réemploi du graphe routé, remplace seulement l'agrégat routed-experts brut, et refuse toute sortie hôte si une couche manque ou échoue. Garder les branches shared/residual et l'application unique des coefficients; compiler et tester le patch donneur sur CPU avant tout chargement de modèle.
