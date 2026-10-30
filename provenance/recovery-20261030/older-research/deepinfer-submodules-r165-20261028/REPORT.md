# R165 — Quatre sous-modules NVIDIA de DeepInfer récupérés

Les sources DeepEP, DeepGEMM, DeepSelect et FlashMLA ont été initialisées aux gitlinks exacts du clone R163 ; références, SHA256 des README et licences et état Git dans manifest.json. Clone principal et quatre arbres propres ; le total est de 20 Mio. L'espace PC6 était d'environ 16 Gio avant téléchargement. Les deux sous-modules Ascend et cinq gitlinks de dépendances imbriqués ne sont pas initialisés. Pas de build, installation, test GPU ni modèle.

Les quatre dépôts déclarent MIT à leur racine ; cela ne préjuge pas des licences de CUTLASS, DeepJIT ou autres dépendances transitives.

Observations nouvelles par lecture ciblée :

- DeepEP fournit `deep_ep/utils/refs.py` avec références de dispatch et combine ; dispatch utilise `torch.distributed.all_to_all_single` et est non expansé. Son combine fait une réduction groupée stable avec accumulateur FP32 et réencodage intermédiaire au dtype des données, selon les modes choisis. C'est un candidat d'oracle comparatif, mais pas la sémantique prouvée du Collector Rust FP16/f64 ni une preuve de fonctionnement inter-PC. Tester séparément sans assimilation automatique.
- DeepSelect vise le top-k d'attention sparse et l'échantillonnage ; ses paramètres de stride, indices hors limites et comportement NaN sont dans `deep_select/interface.py`. Il ne faut pas le confondre avec le gate MoE de DeepInfer qui appelle TileKernels. Compatibilité V100 non établie.
- FlashMLA épinglé déclare NVIDIA SM100/SM103. Son README précise que cette version a changé le format du cache KV FP8/FP4 et retiré la prise en charge Hopper et de modèles antérieurs. Une adaptation doit fixer précisément la version et le format ; aucun support V100 n'est déclaré.
- DeepEP et DeepGEMM gardent les limites V100 et transport documentées en R164. Les sources locales sont désormais disponibles pour leur dissection ciblée ; aucune qualité ni performance n'est transférée automatiquement.

Prochaine tranche : construire une comparaison CPU bornée et indépendante du combine DeepEP et du contrat Collector choisi, notamment les arrondis et l'ordre, tout en poursuivant le pont Session–propriétaire réel. Les graphes CUDA, le modèle complet, le réseau de production et la génération ne sont pas qualifiés.
