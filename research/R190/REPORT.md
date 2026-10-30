# R190 — huit routes et deux propriétaires CPU

Huit copies synthétiques du même expert Qwen échantillonné R132 sont emballées avec le CLI Rust en deux artefacts résidents, quatre experts distincts par rang (2 et 9). Leurs codes et échelles proviennent tous de la même extraction vérifiée : ce ne sont pas huit vrais experts du checkpoint. Les fichiers sont des données de test locales et ne doivent pas être ajoutés à un commit.

La vraie Session Rust et deux `PersistentQPN2Owner` utilisent un fournisseur CPU de test généralisé ; deux batches d'un token, huit routes distinctes et coefficients 1/8 passent. Chaque rang charge une fois quatre experts puis réalise deux EXEC ; les 2 048 résultats des deux batches correspondent à la ligne FP16 de référence. Run `r190-415ad144087043d7966dcd2105c5f139`, sortie SHA256 `393bb48770c6b746b089f29e42ff7172dd4ef8e07743535d9648a0979271c904`. `test_eight.py` : 1 test, exit 0.

Portée limitée à une entrée, huit experts synthétiques, alignement et fournisseur CPU de test ; aucune qualification modèle complet, GPU ou performance. Les artefacts sont retenus localement sous `artifacts/` et leurs empreintes sont dans `artifacts/manifest.json`.
