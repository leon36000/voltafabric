# R148 — sélection du dtype automatique, source 1Cat et simulations CPU

Les membres `vllm/config/model.py` et `vllm/platforms/cuda.py` sont liés par SHA au wheel 1Cat 1.2.2 R125 relu sur PC1. Le code source du wheel indique que `dtype` vaut par défaut `auto` et que le choix automatique privilégie le premier dtype supporté lorsque celui du modèle n'est pas supporté. Pour les cartes de capacité entre 6.0 et moins de 8.0, `supported_dtypes` renvoie FP16 puis FP32 ; BF16 n'est pas dans cette liste.

Quatre tests CPU extraient et exécutent par AST les fonctions exactes de ce wheel avec une plateforme et des types simulés. Pour un modèle Qwen et une configuration BF16 simulée, `auto` choisit FP16 sur SM70 et conserve BF16 sur SM80 ; FP16 sur SM70 est conservé et FP32 sur SM70 est réduit à FP16. `python3 -B test_dtype.py` : 4 tests, exit 0. Les sources extraites restent dans `.runs/r148-dtype-source`, leurs SHA sont dans le reçu.

Cela précise R147 : on ne peut pas déduire un repli Triton du seul champ BF16 du checkpoint, car le dtype automatique peut devenir FP16 sur V100. Combiné aux tests synthétiques R147, ce résultat rend la branche FlashQLA plausible si ses autres conditions sont satisfaites. Aucune commande réelle de lancement, conversion effective de config, bibliothèque, GPU ou modèle n'a été exécutée ; le dtype réel reste à observer. Une sélection explicite du dtype change le raisonnement. Les contraintes 1Cat/TileLang restent contradictoires.

Prochaine action : établir le dtype issu du convertisseur du checkpoint et les arguments de lancement effectifs, puis valider la frontière MoE sur CPU avant l'admission d'un environnement GPU isolé.
