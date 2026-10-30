# R167 — contre-exemple R166 dans le Collector Rust réel

Le CLI `collect-fp16` existant, SHA256 `11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4`, a reçu trois lignes FP16 synthétiques `[2048,1,-2048]`, experts distincts 0,1,2 sur deux rangs, coefficients unitaires, avec un nouveau stamp CPU. Trois lignes acceptées : résultat **1** à l'arrivée normale et à l'arrivée inversée, conformément à l'ordre des slots du Collector. Une ligne omise provoque `BATCH_INCOMPLETE`, exit 2 et aucune sortie partielle. Entrées et reçu complets dans ce dossier, `python3 -B run.py`, exit 0.

La comparaison avec R166 est limitée à ces contributions et à cette configuration synthétique : la référence DeepEP de réduction locale donnait **0** à cause d'un arrondi FP16 intermédiaire ; sa variante sans réduction locale donnait **1**. Les coefficients 1 isolent l'arithmétique et ne prétendent pas représenter la distribution d'un gate réel. Ceci démontre une incompatibilité numérique de ce mode de référence avec le contrat du Collector ordinaire sur cet exemple, non une erreur du donneur ou de son kernel. Pas de GPU, service, exécution modèle, communication réelle ni bridge Session–owner.

Suite : examiner une adaptation explicite préservant l'arithmétique choisie, et raccorder puis tester le pont CPU Session–owner avec refus atomique et UNKNOWN jamais rejoué.
