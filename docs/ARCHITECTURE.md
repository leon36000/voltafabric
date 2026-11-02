# Architecture du socle CPU — 0.1.0

## Frontière produit

Cette bibliothèque est un socle exécutable de contrats, références et planification.
Elle ne sert pas encore un vrai LLM. Skinny et son moteur restent la cible d'intégration.
Rust porte les structures, la validation et les algorithmes. Python porte uniquement
l'oracle indépendant et l'adaptateur de propositions LiteLLM. Aucun kernel GPU n'est
remplacé par une grande inférence CPU prétendument compétitive.

## Modules présents

| Module | Contrat | Limite |
|---|---|---|
| memory | Arithmétique u64 vérifiée, blocs16, plans alignés, pic de chargement | Le layout réel du checkpoint doit être vérifié séparément |
| quant | E2M1, échelles E4M3 non négatives, multiplicateur global de déquantification | Référence linéaire CPU, pas layout QPN2/TE ni kernel SM70 |
| moe | Référence séquentielle, workers CPU par rang, mapping global/local | Mémoire partagée locale, pas transport réseau |
| Collector | Identité run/modèle/plan/couche/batch/epoch, résultats complets ordonnés | Contrat logique, pas authentification réseau ni exactement-une-fois |
| topology | Liens orientés, ressources partagées, profil PCIe génération/largeur facultatif | Le profil borne les débits PHY annoncés; aucune détection ou preuve automatique du P2P |
| traffic | Codes, headers, métadonnées, aller-retour, précision de sortie | Calcul d'octets, pas temps réel |
| schedule | DAG et ressources exclusives, validation des cycles | Ordonnanceur de listes, pas simulateur de partage fluide de bande passante |
| planner | Best-fit puis déplacements/échanges améliorants et bornés; API facultative de scénarios concurrents | Scénarios explicitement fournis, ressources GPU/liens exclusives, makespan analytique; pas de partage fluide de bande passante ni mesure réelle |
| pipeline | DP de coupes contiguës pour ordre de stages fixé | Optimal seulement pour ce modèle additif, pas parallélisme général |
| vf | CLI JSON bornée, erreurs explicites, capacités déclarées | Pas de daemon, RPC, API CUDA, distribution réelle |
| deepseek_v41_sampler | Argmax greedy CPU à identité complète, digest tokenizer, EOS et budget de complétion; argmax à égalité stable | Prend les logits du backend; aucun LM head, chargement de checkpoint ou génération complète |

The sampler adapts only the deterministic ordered-candidate argmax rule documented in
llama.cpp's `llama-sampler.cpp` at commit
[`29c720eaf72fcf8e4e61a56a88b994d92668261d`](https://github.com/ggml-org/llama.cpp/blob/29c720eaf72fcf8e4e61a56a88b994d92668261d/src/llama-sampler.cpp#L1054-L1061).
That upstream revision is MIT-licensed; no llama.cpp source code was copied into this
crate. The VoltaFabric adapter adds checkpoint tokenizer/EOS and request-identity
admission around the argmax policy.

Le planner ne modifie pas les choix d'experts ni leurs poids. Il n'alloue qu'un
propriétaire par expert; la réplication du registre MoE est testée séparément.
`plan_with_concurrent_scenarios` et `score_with_concurrent_scenarios` acceptent des
indices vers les demandes exactes déjà présentes. Chaque scénario ajoute son poids
au score et planifie les DAG des demandes ensemble avec des identifiants stables.
Le modèle partage les ressources GPU et lien comme exclusives; ce n'est pas un
simulateur de bande passante fluide ni de files d'attente. Les scénarios sont
bornés et gardent le contrôle de poids résidents. L'agrégation du trafic est
sélectionnée sur les octets et non sur une précision numérique supposée équivalente.
Le score ne comprend pas encore le coût d'une réduction locale. Tout optimum
matériel reste à mesurer.

Un lien `pcie_p2p` peut fournir `pcie: { generation, lane_width }`. Le validateur
rejette un débit déclaré supérieur au plafond PHY calculé à partir du débit de
signalisation et de l'encodage de ligne. Ce plafond exclut le protocole, les FLIT,
la FEC, les copies et les coûts logiciels : ce n'est pas un débit utile ni une
mesure. Gen6/7 sont bornés par le signal PAM4 brut avant les surcoûts FLIT/FEC.
Leurs plafonds reprennent les débits bruts PCI-SIG de 64/128 Gbit/s par voie
(128/256 GB/s unidirectionnels en x16); le facteur PAM4 de deux bits/symbole ne
doit pas être appliqué une seconde fois ([Gen6](https://pcisig.com/faq?field_category_value%5B%5D=pci_express_6.0&keys=PAM4),
[Gen7](https://pcisig.com/what-data-rate-pcie-70-specification)).
L'absence du profil reste acceptée pour la compatibilité; les identifiants et
largeurs négociés doivent être relevés séparément et ne prouvent pas un P2P actif.

## Identité et erreurs

Une réponse appartient à un tuple complet. Résultat manquant, doublon, ancien epoch,
expert local erroné, dimension invalide, valeur non finie ou expiration invalident
le collecteur. Pas de sortie partielle, pas de renormalisation pour cacher l'absence.
Une future reprise devra utiliser une nouvelle identité et vérifier les reçus avant
réexécution. Le collecteur actuel n'est pas un protocole de reprise distribué.

## Exports et simplicité

Entrées et sorties JSON version1. Les structs d'entrée rejettent les champs inconnus.
`vf` limite l'entrée à 16 Mio; les calculs de référence/planning sont bornés. Les
workers LiteLLM ont des limites distinctes et ne disposent d'aucun outil d'exécution.
Ne pas exposer la CLI comme service multi-tenant sécurisé sans couche supplémentaire.

## Intégration future

Définir d'abord un adaptateur minimal pour le mapping exact de Skinny. CUDA/C++ est
acceptable pour les kernels, Rust pour leurs contrats. Une frontière FFI doit rester
petite, testable, avec ownership et synchronisation explicites. Le `forbid(unsafe_code)`
du crate actuel n'est pas une affirmation sur tous les crates transitifs ni un
prétexte pour contourner une FFI indispensable dans un crate séparé.

Réutiliser un moteur corrigé TP+PP comme baseline. EP, FlashMoE, NCCL Device API ou
MSCCL++ sont des candidats, pas des dépendances toutes obligatoires. Aucun backend
n'est qualifié par le simple fait qu'une documentation amont annonce SM70.

