# Dépendances, sources et droits

## Ce qui est incorporé

Le crate Rust ne copie aucun kernel tiers : uniquement serde/serde_json via Cargo,
avec Cargo.lock pour la résolution testée. Les crates ne sont pas vendoriés : une
première installation a besoin du registre ou d'un cache préexistant. « Sans V100 »
ne signifie pas « distribution intégralement déconnectée ». Rust1.90.0 est figé.

Les scripts actifs utilisent Python standard library. La preuve NumPy/pytest héritée
est séparée dans `reference/prior_qualification/`; elle n'est pas une dépendance du
chemin principal. Les registres de recherche sont historiques, pas tous re-audités
pendant cette construction. Le propriétaire doit choisir une licence de publication
avant diffusion publique; aucune forge ni publication publique n'est créée ici.

## Sources officielles relues pour le handoff, 7 octobre 2026

- OpenAI, instructions et contexte ciblés plutôt qu'accumulation de règles :
  https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra
- OpenAI, découpage des rôles et portes de validation multi-agent :
  https://developers.openai.com/cookbook/examples/codex/codex_mcp_agents_sdk/building_consistent_workflows_codex_cli_agents_sdk
- OpenAI, boucle agentique et gestion du contexte :
  https://openai.com/index/unrolling-the-codex-agent-loop/
- LiteLLM, budgets, réservations et dépendance à la base de données :
  https://docs.litellm.ai/docs/proxy/users
- LiteLLM, limites et provenance des coûts :
  https://docs.litellm.ai/docs/proxy/cost_tracking
- NVIDIA, NVFP4 (bloc16, échelles, format et swizzling) :
  https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html
- Rust, installation officielle : https://www.rust-lang.org/tools/install

Ces sources soutiennent des techniques, pas une supériorité universelle du prompt.
Aucune comparaison expérimentale des prompts sur tous les modèles n'a été effectuée.
Les docs évoluent; Codex doit vérifier les versions réellement installées, sans
réécrire l'environnement simplement pour rejoindre la dernière documentation.

## Réutilisation future

Utiliser SOURCE_PINS.json hérité comme point de départ, vérifier le commit vivant,
fichiers exacts et licence avant extraction. Attention particulière : restrictions
TCCL, hypothèses matérielles FlashMoE/cuBLASDx, licences/dépendances des transporteurs,
format quantifié réel de Skinny et chemins EP explicitement supportés. Conserver
notices/licences lors de l'incorporation. Un README SM70 n'est pas une preuve binaire.

## Emprunts intégrés, état vivant vérifié le 8 octobre 2026

Source : https://github.com/gilby/volta-nvfp4, commit
b1a2181702e609dc92591343e46c63d1324ade43. Le clone public isolé a été lu, les fichiers
et HEAD hashés ; le registre historique n'est pas la preuve de cet état.

| Fichiers amont | Usage ici | Droits conservés | Compatibilité/adaptation |
|---|---|---|---|
| fork_patches/skinny_moe.py | adapters/skinny_moe.py | Apache-2.0, notices tierces | ABI QPN2 existante ; compaction locale, masque ownership, combine FP32, validations ; Torch CPU testé, vLLM/GPU encore non qualifiés |
| fork_patches/marlin.py, fonction _qpn_prepack | tests/reference/qpn_prepack.py | Apache-2.0 | Fonction amont inchangée extraite pour référence Torch ; codec Rust indépendant, pas de remplacement du moteur |
| kernels/qpn2_moe_kernel.cu et helpers kernels/skinny_kernels.cu | cuda/qpn2_moe_core.cuh | MIT pour kernel/MMA/fp8 ; Apache-2.0 pour dequant8_tm dérivé TurboMind | Extraction device exacte, wrapper C ABI v1 ajouté ; compilation SM70 seule, exécution pending |
| Mapping de fragments _qpn_prepack | src/qpn2.rs | Apache-2.0 conservateur | Permutations Rust bytes, erreurs structurées, travail CPU borné |

Les textes et attribution explicite sont sous third_party/volta-nvfp4 et
cuda/third_party. Le helper TurboMind n'est pas requalifié MIT à partir de la licence
générale du dépôt. Aucun composant restrictif ignoré n'est incorporé au futur service.

Checkpoint : https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4,
révision 1355db6a052410cfd62085d94b58866fd0f2c3c5, Apache-2.0 déclaré par NVIDIA.
Producteur ModelOpt0.44.0. Sources config/hf_quant_config/index/header et ranges
tracés ; uniquement 165900 octets de poids/scales préfixes récupérés, puis trois
crops 64×64 conservés. Safetensors0.6.2 réexporte/recharge les plans compressés ;
Torch2.10.0+cpu sert d'oracle float8. Ce sample n'est pas le modèle entier.

Contrat des crops : codes U8 row-major low nibble puis high nibble, groupes16 ;
scales F8_E4M3 ; global_dequant est la valeur F32 little-endian weight_scale_2
multiplicative, distincte pour gate/up/down, sans inversion ou fusion implicite.
Toutes les dimensions du crop sont alignées ; aucun padding caché. Le QPN2 stocke
[E,N/32,K/16,32,8] codes et [E,N/32,K/16,32] scales en ordre de fragments. Les
permutations conservent des bytes opaques, sans conversion flottante des poids.
La FFI cible le donneur SM70 et sa lecture uint2 little-endian. Les buffers sont
empruntés jusqu'à completion, les tableaux Rust et l'export possèdent leurs bytes.

Toolchain CUDA isolée : redistrib officiel12.8.1, nvcc12.8.93, cudart/cccl/cuobjdump
12.8.90 et GCC14.3 officiels extraits sans install système. 136379112 octets téléchargés,
SHA vérifiés. Six noexcept sur déclarations seulement pour libc de cet hôte :
patch, original et hashes dans evidence/continuation_a123/cuda/toolchain-compat.json.
Le rebuild exige cette même toolchain locale ; elle n'est pas vendoriée dans le pack.
Le coût de maintenance du wrapper reste conditionné à H1, sans promesse de débit.
