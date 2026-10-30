# R161 — sorties FP16 du propriétaire vers collecteur Rust

Les octets FP16 des deux sorties nominales du vrai propriétaire CPU R158 ont été relus hors ligne avec leur reçu SHA. Le test a reconstruit les métadonnées d'identité à partir du routage associé et a transmis les bits binary16 au vrai CLI Rust `collect-fp16`, sans les convertir en tableaux f64 de complétion côté Python. Le collecteur accepte deux lignes, 8 192 octets FP16, et les 2 048 valeurs retournées sont exactement égales à la référence scalaire `0,75 × ligne` : les coefficients de routage sont donc appliqués une fois dans ce cas.

Cinq essais distincts avec attempt erroné, identité locale erronée, doublon, valeur FP16 infinie et ligne manquante sont chacun refusés par le CLI (exit 2, `ok=false`). Le nominal et le script de contrôle sortent 0. Plan externe, SHA des entrées et sorties et messages de refus sont conservés. Les tentatives sont des corruptions de copies CPU ; aucun ancien worker ou UNKNOWN n'a été rejoué.

Ce lot vérifie la couture de représentation FP16 et le collecteur avec les octets capturés d'owners réels. Les métadonnées ont été reconstruites hors ligne : cela ne qualifie pas le transport ou une Session connectée au propriétaire, ni un calcul GPU ou un modèle complet. Les deux experts restent des copies synthétiques d'un seul échantillon.

Suite : raccorder les vrais messages Session au propriétaire sous un contrat FP16 explicite et tester le cycle de vie et les échecs ; préserver la distinction avec l'oracle f64 R160.
