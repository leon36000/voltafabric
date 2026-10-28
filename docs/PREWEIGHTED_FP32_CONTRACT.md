# R11 : collecte des sorties FP32 déjà pondérées

État du raccord : le worker CPU R12 négocie désormais ce format ; R14 le raccorde
également au worker GPU natif explicite décrit dans `../gpu-worker/README.md`.
Le texte ci-dessous conserve le périmètre initial R11. Son absence de worker
négociant le format décrit cette étape historique, pas l'état courant R14.

`preweighted::Collector` reçoit les sorties du laboratoire R9, où le coefficient
de routage est appliqué avant l'arrondi BF16, le packing des activations et W2.
Déplacer cette multiplication après W2 change le calcul. La refaire au collecteur
pondère deux fois. Les anciennes API `moe::Collector` et FP16 restent non pondérées
en entrée ; leurs contrats et ceux des workers persistants ne changent pas.

Le nouveau type `Completion` transporte les bits FP32, les bits du coefficient
original, l'identité complète `Stamp`, token/slot/expert/rank/local et un contrat.
Le contrat porte le format `r9_mxfp4_preweighted_fp32_slot_sum_v1` et le SHA-256
de la sérialisation structurée du dispatch validé et du domaine de version.
Il lie notamment les coefficients et l'ordre physique des experts résidents.
Ce digest vérifie l'accord des métadonnées ; il n'authentifie pas un producteur
et ne prouve ni le calcul GPU, ni la synchronisation, ni les poids employés.

Les coefficients doivent être exactement représentables en FP32 : l'aller-retour
FP64→FP32→FP64 conserve tous leurs bits, signe de zéro compris. Ainsi `0.1` FP64
est refusé, tandis que la valeur FP64 exacte de `0.1f32` est admise. Les neuf cas
R9 conservés sont admissibles ; la couverture d'un routeur de modèle complet
n'est pas mesurée. Aucun arrondi silencieux du coefficient n'est autorisé.

La somme suit l'ordre original des slots, indépendamment de l'ordre d'arrivée,
avec un arrondi FP32 à chaque addition, depuis +0. Ce contrat de laboratoire
n'affirme pas reproduire l'ordre d'un kernel distribué amont. Il normalise un
résultat zéro en +0. Un coefficient nul n'admet que des valeurs ±0. NaN, Inf,
dimensions ou identités incorrectes, doublons, expiration et contrat différent
font échouer le batch ; une somme débordante ou un résultat manquant ne produit
aucune sortie de batch. Le collecteur conserve les limites d'allocation existantes.

Les commandes `vf preweighted-fp32-contract plan.json` et
`vf collect-preweighted-fp32 batch.json` exercent cette frontière sur CPU, avec la
limite JSON existante de 16 MiB. La seconde renvoie `fp32_bits` et
`scope=CPU_RECEIPT_COLLECTION_ONLY`. Ce JSON n'est pas un nouveau transport de
service. Aucun worker existant ne négocie encore ce format.

Validation reproductible :

```sh
cargo +1.90.0 test --locked --test preweighted_fp32
cargo +1.90.0 test --locked --release --test preweighted_fp32
python3 -B scripts/preweighted_crosscheck.py --binary "$CARGO_TARGET_DIR/release/vf"
python3 -B scripts/verify.py --run-id r11-reproduction-unique
```

Employer la toolchain/cache isolés du handoff et un identifiant de run neuf.
L'oracle Python utilise des fractions exactes et un encodeur IEEE-754 indépendant.
La qualification R11 conserve également les refus CLI, six mutations compilables
et les comparaisons PC6 ARM64 / PC1 x86-64 dans
`../.runs/preweighted-collector-20261024-r11/`. Aucune exécution GPU, mesure réseau
ou qualification d'un service complet ne découle de ces tests CPU.
