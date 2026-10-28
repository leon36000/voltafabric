# R107 — modèle complet et installation existante retrouvés sur PC1

**Le blocage de localisation R105 est résolu : un modèle texte Qwen3.8 SC117, ses deux GGUF, son pack Strata, son tokenizer et son binaire configuré existent sur PC1. Ne pas installer ni télécharger un doublon.** Ce relevé établit des chemins et une cohérence de métadonnées, pas une vérification des payloads ni une génération actuelle.

Installation sélectionnée par `/home/pc1/Strata/strata-sc117-abliterated-iq3_s.json` :

- Modèle déclaré : `qwen3.8-flash-next-sc117-abliterated-iq3_s`.
- Moteur : `/home/pc1/Strata-data/optimizations/2026-10-14/build-zen5/strata`, fichier présent de **75892384 octets**.
- BUILD adjacent : **Strata 0.1.44.1**, source déclarée `296b59d2f73c8f04a805df29c2fa97655adace78`, architectures compilées `[120]`, flags CPU Zen5. Ce BUILD n'est pas un hash du binaire et ne prouve pas sa correspondance effective.
- Pack : `/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s` : `experts.bin` **50292326400 octets**, `dense.bin` **1538035200 octets**, `index.txt` **93697 octets** / 1079 entrées déclarées, `native_experts.txt` **3316 octets**, 48 lignes de couches plus en-tête.
- Tokenizer : `/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s/tokenizer` : `vocab.json` **5737005 octets**, `merges.txt` **3353258 octets**, `tokenizer.json` **541 octets**, `token_type.json` **744960 octets**, `chat_template.jinja` **9993 octets**.
- Profil experts référencé : `/home/pc1/Strata/data/expert-profile.bin` ; son existence/format n'a pas été vérifié dans ce lot.

GGUF originaux, tous deux présents dans `/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S` :

| Fichier | Octets | Tenseurs déclarés |
|---|---:|---:|
| `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf` | 55063448064 | 1223 |
| `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf` | 28800138432 | 1 |
| Total | **83863586496** | **1224** |

Le deuxième shard est aussi le `--ple-gguf` de la configuration. Headers GGUFv3 : architecture `qwen4exp`, 48 couches, embedding2560, 512 experts dont10 utilisés, contexte déclaré262144. Tokenizer GPT2/pre `qwen35`, vocabulaire248320, merges247587, EOS248046, BOS248044. Ces nombres concordent avec le petit tokenizer.json du pack ; la longueur du template concorde aussi. Les deux descriptions source du pack nomment exactement les deux shards et les mêmes tailles. Les couches0..47 sont présentes dans l'index. La correspondance numérique des poids/embeddings/attention n'est pas démontrée par ces contrôles.

Le checkout `/home/pc1/Strata` a un HEAD `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, et son ancien `engine/BUILD.json` annonce0.1.41. **Ne pas confondre ces identités avec le binaire configuré0.1.44.1 ni avec l'archive R46/R67 `4b9e05a…` utilisée par R105.** Le précontrôle R105 GGUF source-only ne valide pas ce pack natif/configuration optimisée tel quel.

## Services déjà actifs et PC6

Au relevé, le PID historique646996 sur PC1 est `llama-server` dans `/home/pc1/.local/share/pc5-rag-accelerator/runtime/llama-server`, avec `-m /home/pc1/.local/share/pc5-rag-accelerator/model/Nemotron-3.5-Embed-8B-Q4_K_M.gguf` (**4896392704 octets**). Il n'est pas Strata. PC6, hôte `gx10-e526` ARM64, possède un `llama-server` similaire PID2885791, exécutable `/home/pc6/.local/share/pc5-rag-accelerator/bin/llama-server`, modèle `/home/pc6/.cache/pc5-mass-ingestion/model/Nemotron-3.5-Embed-8B-Q4_K_M.gguf`, même taille. Le nom des modèles indique l'usage embeddings ; ce ne sont pas des preuves d'un service de génération de réponses. Aucune requête ne leur a été envoyée.

Aucun processus dont le nom est Strata n'a été observé dans le relevé ciblé. Cela n'est ni un inventaire GPU ni une réservation de PC1. Les services existants restent intacts. Les manifests d'assets llama existent, mais le filtre prudent n'en a pas extrait une version ; la version exacte de ces deux installations reste non établie.

## Méthode, preuves et limites

STORAGE_INDEX/R84 ont fourni les racines et confirmé l'absence d'un déplacement de poids. Le SSH réutilise exactement l'alias `pc1`/BatchMode/StrictHostKeyChecking des reçus R99/R80 ; aucune nouvelle route/clé. Les seules listes de répertoires sont aux chemins directement découverts. Les processus ont livré exe/cwd et uniquement des flags de chemins modèle/configuration autorisés. Aucune ligne de commande brute, environnement de processus, credential, log de conversation, HTTP, signal ou attachement n'a été utilisé.

Lectures des seuls headers GGUF : **3970242 octets réellement lus**, curseur jusqu'à10946733 sur shard1 et106 sur shard2, limite16MiB par header. Aucun tenseur/payload de poids n'a été lu ou hashé. Petits JSON/index/template hashés ; stat des poids/binaires. Aucun moteur, GPU/NVML, génération, build, installation, téléchargement ou namespace. Les résultats et sources des sondes sont dans ce sibling ; les versions intermédiaires de `paths_probe` ont été conservées avec leurs hashes exacts sans rejouer les sondes.

Prochaine commande concrète **locale, CPU, sans réseau** :

```sh
python3 -B /home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/full-model-local-discovery-r107-20261028/verify_discovery.py
```

Compatibilité VoltaFabric actuelle : **pas de consommation directe livrée de ce pack Strata**. Le fichier natif référence des types GGUF IQ3/IQ2/IQ4/Q2_0, alors que la frontière VF qualifiée consomme ses artefacts MXFP4 safetensors et ses contrats propres. Une simple égalité de dimensions ne rend ni formats, ni activations, ni placement des coefficients équivalents. La compatibilité numérique d’un adaptateur complet reste inconnue ; aucune conversion ou intégration n’est livrée ici.

Le prochain travail utile est une admission CPU **du pack/configuration existants de PC1** : parser l'index natif et ses extents contre les fichiers présents, vérifier le profil experts, résoudre la provenance exacte du binaire0.1.44.1 et les hashes de payload déjà disponibles, vérifier vocab/merges/template avec le tokenizer de cette version. Aucune recommandation de charger ou de reconfigurer le service actif. Une éventuelle génération de contrôle upstream et l'intégration VoltaFabric resteront deux preuves séparées sous leurs gates.
