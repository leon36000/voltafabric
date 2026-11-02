# R700 — ciblage arXiv pour VoltaFabric

Portée : une passe de repérage large effectuée par un sous-agent GPT-6 Luna,
suivie d'un tri des pistes liées à la résidence des experts, au préchargement
et au KV. Ce fichier conserve des pistes vérifiables; ce n'est pas la preuve
d'une lecture exhaustive de tout arXiv. Aucun résultat de performance n'est
transposé au parc V100/PC4/PC6.

| Travail | Idée à retenir | Vérification logicielle/licence | Décision pour VoltaFabric |
|---|---|---|---|
| [DAOP, arXiv:2501.10375 v3](https://arxiv.org/abs/2501.10375) | Placement dynamique d'experts selon les activations et pré-calcul CPU sélectif des experts prédits afin de réduire les transferts. | Dépôt [ecolab-nus/DAOP](https://github.com/ecolab-nus/DAOP), commit `08fef3330fe73249b757d26c2c00cd0dfecb840d`, licence Apache-2.0 vérifiée au dépôt et dans `LICENSE`. Le README le qualifie de preuve de concept en construction et rapporte des essais A6000. | Garder l'idée de prélecture sous budget; ne pas reprendre ce runtime ni ses chiffres. VoltaFabric a déjà un conseil de mémoire par rang à relier à des routes réelles. |
| [LayerScope / PreScope, arXiv:2509.23638 v2](https://arxiv.org/abs/2509.23638) | Le papier LayerScope décrit PreScope: prédiction d'activations par couche, ordonnancement global de préchargement inter-couches et I/O asynchrones. | Aucun dépôt ni licence de code n'ont été établis dans cette passe. | Candidat conceptuel pour planifier un préchargement; mesurer d'abord les transferts et vérifier la prédiction sur les routes du modèle cible. |
| [ELDR, arXiv:2607.00466 v2](https://arxiv.org/abs/2607.00466) | Utilise une signature d'experts observés en préfill pour router les requêtes vers des workers de décodage selon la localité prédite et leur charge. Le papier décrit une implémentation vLLM. | Dépôt et licence effective non vérifiés dans cette passe. L'hypothèse porte sur une flotte de workers, ce qui dépasse l'exécution locale qualifiée actuelle. | À différer jusqu'à un flux distribué réel; ne pas en déduire une politique qui enfreindrait l'unicité `(expert, rang)` du registre VoltaFabric. |
| [PiKV, arXiv:2508.06526 v3](https://arxiv.org/abs/2508.06526) | Stockage KV shardé par experts, routage vers les entrées pertinentes, ordonnancement et compression du cache KV. | Dépôt [NoakLiu/PiKV](https://github.com/NoakLiu/PiKV), commit `36666b925293d46c6da7611aecb2b21547b78d2a`: métadonnée de licence API nulle et aucun `LICENSE` à la racine à ce commit; le README revendique Apache-2.0. Licence du code donc non validée à ce snapshot. | Ne pas copier pour l'instant. Reconsidérer si la bande passante KV devient un goulot mesuré et que le snapshot possède un texte de licence vérifiable. |

## Triage global

DAOP est la seule piste de cette sélection avec une licence de dépôt vérifiée,
mais son état de preuve et son matériel d'évaluation ne suffisent pas à en faire
un composant qualifié pour VoltaFabric. LayerScope/PreScope et ELDR restent des
idées d'ordonnanceur; PiKV est bloqué pour la réutilisation de code par la
licence. Le prochain code doit s'appuyer sur les frontières déjà présentes:
route gate exacte, budget mémoire, poids résidents, identité complète et refus
atomique. La phase de distribution inter-PC demeure suspendue aux admissions et
aux preuves dédiées; aucun transport/RDMA n'est qualifié par cette revue.
