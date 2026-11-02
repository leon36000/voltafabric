# R738 — route MXFP4 directe par Sessions résidentes

## Changement

R737 composait attention fenêtrée et FFN, mais la route top-6 passait par une fermeture `FnOnce` qui devait elle-même attester la résidence. L'API ajoutée ici relie le plan produit par le gate à `Session` et au `Collector` VoltaFabric existants. Elle ne charge aucun poids sur le chemin token.

Pour chaque token, le bloc prend le jeu de Sessions par couche, convertit l'activation gate BF16 exacte en entrée de protocole, valide la route et l'identité (modèle, couche, epoch, batch/attempt), puis préflight toutes les Sessions participantes avant le premier `EXEC`. Le préflight reprend les validations du Session sur artefact chargé, rang, couche, epoch, hidden, ordre physique des propriétaires locaux et slots sélectionnés. Les appels `start` partent avant la collecte; le Collector FP32 prépondéré existant applique les coefficients une seule fois. Une sortie n'est publiée qu'avec six slots complétés et une ligne entière. `Aborted`/`Unknown` expire le Collector; aucune relance n'est faite et le cache KV externe ne progresse qu'après la réussite du bloc complet.

## Réemploi et provenance

R738 réutilise directement `Session::preflight_preweighted_fp32_until`, `start_preweighted_fp32_until`, `finish_preweighted_fp32`, `Collector`, la route privée déjà produite par le gate et la composition cache atomique de R737. Il n'intègre ni ne copie de code vLLM, TensorRT-LLM, llama.cpp, SGLang, Maya ou Strata. DeepSeek-V4.1-Flash reste la référence sémantique MIT épinglée en R737.

## Limites

Cette frontière vérifie l'état chargé connu du `Session` et son binding de dispatch; le chargement initial doit toujours provenir du plan/checkpoint vérifié et la correspondance avec les fichiers officiels n'est pas ré-auditée token par token. Aucun poids réels du checkpoint, oracle numérique indépendant, tests manuels, GPU, transport inter-PC, logits/tokenizer, échantillonnage ou génération multi-token n'a été exécuté ici. Un `Session` chargé doit pouvoir contenir tous les experts locaux annoncés par `resident_order`; la capacité et le placement réels restent à qualifier.

## Vérification

La source sera soumise aux contrôles automatiques de la branche privée. Aucun test n'a été ajouté ni lancé manuellement dans cette tranche. La preuve de compilation, formatage et pipeline sera ajoutée au reçu après lecture exacte du commit et du pipeline.

## Suite

R739: relier une admission de propriétaires issue d'un checkpoint compatible à ces Sessions persistantes, puis confronter une sortie de bloc à un oracle indépendant avant d'élargir la génération. Toujours CPU-only tant que le gate matériel requis n'est pas renouvelé.
