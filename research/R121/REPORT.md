# R121 — admission statique du paquet 1Cat-vLLM

Le wheel public 1Cat-vLLM 1.2.2 cp312 x86-64 épinglé par le bootstrap Skinny a été téléchargé une seule fois dans la Zone Tempon sur PC4 : `/srv/zone-tempon/r121-1cat-wheel-20261028/`. Ses 245 787 536 octets ont été vérifiés contre le SHA-256 attendu `8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6`. Le wheel n'est pas intégré à ce dépôt ; la Zone Tempon n'est pas une sauvegarde. Il n'a pas été installé ou exécuté.

L'inspection de son ZIP et du bootstrap donneur épinglé révèle une frontière manquante :

- Le wheel contient `modelopt.py` et `marlin.py`, mais aucune entrée portant le nom `skinny_moe`. Leurs SHA sont liés dans `pc4-evidence.json`.
- Le `modelopt.py` du donneur importe explicitement `vllm.model_executor.layers.quantization.skinny_moe`. Le bootstrap épinglé déploie plusieurs patches, dont `modelopt.py` et `marlin.py`, mais ne mentionne pas le déploiement de `skinny_moe.py`.
- Le wheel déclare `tilelang==0.1.9` et `apache-tvm-ffi==0.1.9`. Le bootstrap du donneur leur substitue intentionnellement 0.1.10 pour les deux et indique un conflit de dépendances. Cette justification est une affirmation du donneur, pas une validation locale.

Un environnement construit automatiquement selon ces seules opérations ne doit donc pas être déclaré apte à utiliser le hook Skinny. Il faut préparer une superposition isolée et explicite du module manquant, conserver les fichiers originaux, vérifier le wheel et chaque patch par hash, puis tester l'import et les appels dans un environnement aux dépendances qualifiées. L'adaptateur VoltaFabric doit conserver ses contrats de dispatch et de Collector ; la simple présence d'un module ne les prouve pas.

Limites : inspection statique seulement ; aucun GPU, aucun import de 1Cat-vLLM, aucun modèle ni benchmark. Les huit objets natifs du wheel n'ont pas été chargés. Le checkpoint MoE compatible et l'environnement d'exécution restent à admettre. Le code canonique et les 45 portes CPU historiques sont inchangés.

Sources : [release 1Cat-vLLM 1.2.2](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.2.2), [bootstrap Skinny épinglé](https://github.com/gilby/volta-nvfp4/blob/b1a2181702e609dc92591343e46c63d1324ade43/scripts/bootstrap-sm70.sh), [modelopt épinglé](https://github.com/gilby/volta-nvfp4/blob/b1a2181702e609dc92591343e46c63d1324ade43/fork_patches/modelopt.py). Voir aussi `research/R118/REUSE_PLAN.md` et `research/R119/REPORT.md`.
