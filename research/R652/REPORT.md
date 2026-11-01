# R652 — isolation du réglage de pinning Strata

## Source examinée

Le donneur Strata est épinglé au commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (licence MIT à ce commit). Son `expert_source.cpp` lit `STRATA_STAGE_PIN` dans l'environnement du processus et traite toute valeur non nulle comme une demande de pinning. Le défaut compilé de cette révision est désactivé. La note de version rapporte une corruption du décodage IQ3_S après un prompt de 4096 tokens quand le pinning était actif, vraisemblablement parce qu'un buffer d'étape était réutilisé avant la fin d'une copie asynchrone.

Sources primaires :

- [code `expert_source.cpp` au commit épinglé](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/expert_source.cpp), notamment le commentaire et la lecture de l'environnement autour de `claim_stage`;
- [notes Strata au même commit](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/DETAILS.md), section `STRATA_STAGE_PIN`;
- [licence du donneur](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/LICENSE).

## Adaptation intégrée

`voltafabric_launch.launch()` construit maintenant un environnement propre pour le processus Strata et y fixe `STRATA_STAGE_PIN=0`. Les variables sans rapport sont conservées. L'environnement du processus appelant n'est pas modifié. Les processus coordinateurs Rust ne reçoivent pas ce réglage et leur protocole VFB3 ne change pas.

Cette tranche adapte une contrainte de sûreté du donneur; elle ne copie ni ne lie son code. Elle évite qu'un shell ou service configure involontairement le chemin expérimental connu comme risqué pour les experts IQ3_S de la cible SC117.

## Vérification

`python3 -B research/R652/cpu-harness/test_strata_stage_pin_sanitization.py -v` — **1 test réussi**. Le test appelle le vrai `launch()` avec 48 sockets locales synthétiques, remplace le lancement des coordinateurs et intercepte `subprocess.Popen`; avec `STRATA_STAGE_PIN=1` dans l'environnement parent, il vérifie que le processus enfant reçoit `0`, qu'une autre variable est préservée, que les 48 descripteurs sont transmis et que l'environnement parent reste à `1`.

`python3 -m py_compile adapters/strata_native_bridge/voltafabric_launch.py research/R652/cpu-harness/test_strata_stage_pin_sanitization.py` — **réussi**.

## Portée et suite

Pas de binaire Strata donneur exécuté, de checkpoint chargé, de génération, de GPU, de réseau inter-PC ou de réservation R53. Le test valide l'argument `env` remis à la frontière `Popen`; il ne valide pas l'exécution du donneur ni une parité de modèle. Le prochain travail reste de compiler le commit Strata exact dans un environnement CPU isolé, puis d'installer son vrai callback de génération dans le lanceur et de comparer les sorties CPU à un oracle indépendant.
