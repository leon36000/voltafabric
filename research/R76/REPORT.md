# R76 — entrée FP8 produite sur CPU, essai natif privé SM70

**ACCEPT privé pour exactitude et réduction des arguments de copies Runtime. Aucune promotion moteur ou performance.** Une seule invocation matérielle V100, deux Owners séquentiels, huit vecteurs BF16 distincts par variante, expert0 réel et coefficient0,375 : 81920 valeurs finales totales exactes contre l'oracle Torch R67 gelé. La comparaison indépendante recharge l'oracle R67 original, sans recalculer un golden depuis le candidat. Ce sont des appels natifs directs ; aucune Session/Stamp/Collector/SSH n'est qualifiée par eux.

| Observation par variante | Baseline | Candidat |
|---|---:|---:|
| Entrées H2D demandées au Runtime |163840B|42240B|
| Poids H2D, chargés une fois/Owner |18800640B|18800640B|
| Sorties D2H |163840B|163840B|
| Statut D2H |64B|64B|
| Copies Runtime observées |38|38|
| Compteur de kernels natif |72|64|
| Compteur API CUDA explicite |195|187|
| Valeurs finales exactes |40960|40960|

Économie d'entrée121600B, soit74,21875%, sur ces huit vecteurs. CUPTI a produit152 enregistrements ENTER/EXIT,9728B,76 appels appariés avec phase/corrélation/type/taille/stream/pointeurs/MXCSR. Le candidat réemploie bien le pointeur de sortie pinned privé comme source H2D compact avant D2H, sur le même stream. La trace décrit des octets demandés à cudaMemcpyAsync, pas les transactions PCIe physiques ou une bande passante. Les compteurs de kernels viennent de l'instrumentation native existante, pas d'une trace indépendante de lancements. Aucun temps instrumenté n'est promu en latence, débit ou gain modèle.

Le producteur CPU R5/R3 et la bibliothèque native sont privés. Source inverse : seule branche d'entrée/include/champs de diagnostic changent ; kernels/admission/ABI112/112/teardown/coefficient et quantification predown restent inchangés. Tous les noms de symboles dynamiques correspondent à la baseline, le symbole CPU statique reste local. Les preuves CPU détaillées sont dans REPORT.cpu-batch.md et le snapshot CPU immuable b31d8ad7…60306e. Aucun fichier canonique fonctionnel R59/370 n'a changé.

Un défaut du guard de laboratoire a été reproduit avant GPU : remplacement du chemin du lease après readiness pouvait laisser le guard verrouiller un autre inode. Une copie privée minimale exige maintenant dev/inode/uid attendus avant lock/inventaire, puis recontrôle avant Popen. Les cinq tests CPU réels incluent le témoin ancien qui exécute sur le remplacement et le candidat qui refuse avant inventaire/child. Le guard gelé ancien n'est pas écrasé. Les délais90/105s et le mode GPU réel sont liés au dispatch ; les tests de fichiers/lease/inventaire simulé restent explicitement CPU, pas des cas matériels.

La seule soumission réelle suit la réservation append-only base+suffixe SAME ledgerR53, le plan externe a6f22ae2…cf9cf, un paquet R63 NOREPLACE et les hashes complets. NativePID43921 est observé vivant avec UID1000, PPID43915 et PGID/SID43921, starttime/cmdline/exe liés ; guard43915 et outer43914 sont également réconciliés. Postflight : tous absents, groupes vides, apps GPU vides, mémoire GPU0MiB/util0 à ce point, lease339746 libre, sources/DSO stables. Les registres natifs signalent liveowners0/cleanuperrors0 après destroy ; cela ne remplace pas un nouveau sanitizer ou une qualification de tout le driver.

Limites de preuve : la trace est process-wide, sans filtre Owner/TID ; le programme coopératif ne soumet que les appels attendus. error_count et overflowed sont exigés zéro par le driver hashé, assertions actives, avant écriture du succès, mais ne sont pas sérialisés sur succès. Le contrôle indépendant ne peut donc pas relire ces deux valeurs après coup. Les données brutes/corrélations et21 corruptions offline refusées sont conservées ; aucun GPU n'a été rejoué pour améliorer ces logs. Le code et les flags réellement exécutés sont gelés, pas corrigés rétrospectivement.

Les erreurs initiales de recette PIC/LTO, mutant unused-function, prédicat trop large d'exports, nom de namespace CPU et tail de packaging sont conservées avec continuations réduites. L'ancienne source, le snapshot CPU et les reçus UNKNOWN de transport restent intacts avec amendements CPU explicites ; aucun UNKNOWN logique antérieur n'est rejoué, aucun nouveau UNKNOWN logique n'est produit par ce cas réussi.

Prochaine tranche indépendante R77 : préparer un petit harness natif C++ réutilisant ces mêmes ABI/poids/oracles/bibliothèques, puis au plus un contrôle ComputeSanitizer memcheck distinct, sans charger/abonner CUPTI. Contrats CPU/loader/identités/guard et preuve du vrai enfant CUDA avant verdict. Le memcheck historique R71 ne qualifie pas ce nouveau candidat. Les fautes du candidat, l'intégration Rust/Session, les performances sans observateur, H1 complet, RDMA/GPUDirect, modèle, qualité, énergie et longue session restent séparés.
