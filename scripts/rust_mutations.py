#!/usr/bin/env python3
"""Bounded compilable mutants on Rust and Python guards; canonical source untouched."""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

SKIP = {
    "target",
    ".tools",
    ".runs",
    ".work",
    ".worktrees",
    "evidence",
    "__pycache__",
    ".ruff_cache",
}

# (id, file, exact clause removed, cargo test selector, invariant from AGENTS.md)
CASES = [
    (
        "R01",
        "src/moe.rs",
        "            && !self.received.contains_key(&key)\n",
        ("moe", "collector_rejects_duplicate_and_stays_failed"),
        "doublon provoque une erreur",
    ),
    (
        "R02",
        "src/moe.rs",
        "        let valid = c.stamp == self.stamp\n",
        ("moe", "collector_rejects_duplicate_and_stays_failed"),
        "ancien plan provoque une erreur",
        "        let valid = !self.stamp.run_id.is_empty()\n",
    ),
    (
        "R03",
        "src/moe.rs",
        "        if self.failed || self.received.len() != self.expected.len() {\n",
        ("moe", "collector_cannot_return_partial_output"),
        "resultat manquant: pas de sortie partielle",
        "        if self.failed {\n",
    ),
    (
        "R04",
        "src/moe.rs",
        "        if !experts.insert(e.id) || e.gate.len() != n || e.up.len() != n || e.down.len() != n {\n",
        ("moe", ""),
        "doublon d expert global",
        "        let _ = experts.insert(e.id);\n        if e.gate.len() != n || e.up.len() != n || e.down.len() != n {\n",
    ),
    (
        "R05",
        "src/moe.rs",
        "    if p.registry.epoch != p.stamp.epoch\n",
        ("moe", "stale_registry_rejected"),
        "identite complete run/modele/plan/couche/batch/epoch",
        "    if p.registry.epoch.checked_add(1).is_none()\n",
    ),
    (
        "R06",
        "src/resident.rs",
        "                || !slots.insert((s.token, s.slot))\n",
        ("launch", "foreign_duplicate_and_out_of_range_slots_never_produce_a_plan"),
        "doublon de slot resident",
        "                || {\n                    slots.insert((s.token, s.slot));\n                    false\n                }\n",
    ),
    (
        "R07",
        "src/resident.rs",
        "            || d.stamp.epoch != self.epoch\n",
        ("resident", "stale_static_identity_hidden_or_order_is_rejected"),
        "ancien plan rejete a la frontiere residente",
    ),
    (
        "R08",
        "src/process.rs",
        "            || inputs.insert(input.token, input.value).is_some()\n",
        # The test A26 added for this mutant (RUST_MUTATIONS.json killed_by); the older
        # selector never sends a duplicate input, so R08 survived it.
        ("process", "worker_rejects_a_duplicate_compacted_input_before_started"),
        "doublon d entree compactee",
        "            || {\n                inputs.insert(input.token, input.value);\n                false\n            }\n",
    ),
    (
        "R09",
        "src/completion.rs",
        "    if expected_attempt == 0 || raw.attempt != expected_attempt {\n",
        ("completion", "routed_output_attempt_is_bound_before_collector_mutation"),
        "sortie FP16 liee a l'essai admis",
        "    if expected_attempt == 0 {\n",
    ),
    (
        "P01",
        "src/planner.rs",
        "                if !unique.insert(id) || !experts.contains_key(id) {\n",
        ("", ""),
        "doublon d expert dans une meme demande",
        "                if unique.insert(id) && !experts.contains_key(id) {\n",
    ),
    (
        "P02",
        "src/planner.rs",
        "            if !topology.has_rank(d.source_rank)\n",
        ("", ""),
        "identite du rang source d une demande",
        "            if d.source_rank.checked_add(1).is_none()\n",
    ),
    (
        "P03",
        "src/planner.rs",
        '        if owners.len() != self.p.experts.len() {\n            return Err(invalid("one owner per expert required"));\n        }\n',
        ("", ""),
        "un proprietaire par expert",
    ),
    (
        "P04",
        "src/cost.rs",
        "            || (self.origin == CostOrigin::ImportedMeasurement && self.source.is_none())\n",
        ("", ""),
        "une mesure importee doit citer sa preuve",
    ),
    (
        "P05",
        "src/memory.rs",
        '    if !a.is_power_of_two() {\n        return Err(invalid("alignment must be a nonzero power of two"));\n    }\n',
        ("", ""),
        "alignement puissance de deux pour le pic de chargement",
    ),
    (
        "P06",
        "src/quant.rs",
        "    if !(64..=MAX_K).contains(&k) || !k.is_multiple_of(64) {\n",
        ("", ""),
        "domaine d echelle SM70",
        "    if MAX_K == 0 {\n",
    ),
    (
        "P07",
        "src/completion.rs",
        '    if raw.fp16_bits.len() != count {\n        return Err(invalid(\n            "compact FP16 output length differs; no padding rows allowed",\n        ));\n    }\n',
        ("", ""),
        "pas de lignes de padding dans la sortie compacte",
        "    let _ = count;\n",
    ),
    (
        "P08",
        "src/checkpoint.rs",
        '    if t.dtype != dtype || t.shape != shape {\n        return Err(invalid(format!("wrong dtype/shape for {name}")));\n    }\n',
        ("", ""),
        "dtype et forme du tenseur selectionne",
        "    let _ = (dtype, shape);\n",
    ),
    (
        "Q01",
        "src/routing.rs",
        '    if experts.len() != p.experts.len() {\n        return Err(invalid("duplicate global expert"));\n    }\n',
        ("", ""),
        "doublon d expert global dans le probleme de routage",
    ),
    (
        "Q02",
        "src/schedule.rs",
        '        if j.depends_on.iter().any(|d| !ids.contains(d)) {\n            return Err(invalid("missing dependency"));\n        }\n',
        ("", ""),
        "dependance manquante: erreur et non ordonnancement partiel",
    ),
    (
        "Q03",
        "src/topology.rs",
        "            || d.reserved_bytes > d.capacity_bytes\n",
        ("", ""),
        "reserve au dela de la capacite du peripherique",
    ),
    (
        "Q04",
        "src/traffic.rs",
        "        || f.partial_output_bytes == 0\n",
        ("", ""),
        "tailles d element strictement positives",
        "        || f.partial_output_bytes == u64::MAX\n",
    ),
    (
        "Q05",
        "src/work.rs",
        '        if limit == 0 || limit > DEFAULT_WORK_LIMIT {\n            return Err(invalid("work limit must be within 1..=1000000"));\n        }\n',
        ("", ""),
        "borne du budget de travail",
    ),
    (
        "Q06",
        "src/pipeline.rs",
        '    if p.stages[m - 1].boundary_after_us != 0.0 {\n        return Err(invalid("last stage cannot have a following boundary"));\n    }\n',
        ("", ""),
        "derniere etape sans frontiere suivante",
    ),
    (
        "Q07",
        "src/launch.rs",
        '    if blocks == 0 || blocks > 65535 {\n        return Err(invalid("launch block count outside FFI contract"));\n    }\n',
        ("", ""),
        "nombre de blocs dans le contrat FFI",
    ),
    (
        "Q08",
        "src/resident_artifact.rs",
        '    if output.file_name().is_none() {\n        return Err(Error::new("INVALID_INPUT", "output path must name a file"));\n    }\n',
        ("", ""),
        "le chemin de sortie doit nommer un fichier",
    ),
    (
        "N01",
        "src/quant.rs",
        "    let x = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(n & 7) as usize];\n",
        ("", ""),
        "niveau FP4: aucun renormalisage cache",
        "    let x = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0][(n & 7) as usize];\n",
    ),
    (
        "N02",
        "src/quant.rs",
        "        (1.0 + m / 8.0) * 2.0_f64.powi(e as i32 - 7)\n",
        ("", ""),
        "biais d exposant E4M3",
        "        (1.0 + m / 8.0) * 2.0_f64.powi(e as i32 - 6)\n",
    ),
    (
        "N03",
        "src/quant.rs",
        "        m * 2.0_f64.powi(-9)\n",
        ("", ""),
        "branche sous-normale E4M3",
        "        m * 2.0_f64.powi(-8)\n",
    ),
    (
        "N04",
        "src/moe.rs",
        "        x / (1.0 + (-x).exp())\n",
        ("", ""),
        "signe de la sigmoide dans SiLU",
        "        x / (1.0 + x.exp())\n",
    ),
    (
        "N05",
        "src/moe.rs",
        "        *a = finite(silu(g) * u)?;\n",
        ("", ""),
        "roles gate et up dans SwiGLU",
        "        *a = finite(silu(u) * g)?;\n",
    ),
    (
        "N06",
        "src/moe.rs",
        "                *y = finite(*y + s.weight * v)?;\n",
        ("", ""),
        "perturbation relative de 1e-9 sur la combinaison ponderee",
        "                *y = finite(*y + s.weight * v * 1.000000001)?;\n",
    ),
    (
        "V01",
        "scripts/vf_agents.py",
        "    if not args.authorized_spend:raise ValueError('live calls require existing authorization and --authorized-spend; verify proxy financial limits first')\n",
        ("", ""),
        "aucun appel paye sans autorisation explicite",
    ),
    (
        "V02",
        "scripts/vf_agents.py",
        "        if low in blocked or low.startswith('.env') or 'secret' in low or 'credential' in low or low.startswith('id_rsa'):\n",
        ("", ""),
        "jamais de secret dans le contexte worker",
        "        if low in blocked or low.startswith('id_rsa'):\n",
    ),
    (
        "V03",
        "scripts/vf_agents.py",
        "            if c.execute('SELECT 1 FROM requests WHERE task_id=?',(task_id,)).fetchone():\n                raise ValueError('task already reserved; inspect receipt, do not silently replay')\n",
        ("", ""),
        "pas de rejeu silencieux d une reservation",
    ),
    (
        "V04",
        "scripts/vf_agents.py",
        "            raise RequestFailure('TRANSPORT_OUTCOME_UNKNOWN',uncertain=True) from None\n",
        ("", ""),
        "une issue de transport inconnue reste incertaine",
        "            raise RequestFailure('TRANSPORT_OUTCOME_UNKNOWN',uncertain=False) from None\n",
    ),
    (
        "V05",
        "scripts/vf_agents.py",
        "            if n>=self.limits['max_requests'] or tokens+max_tokens>self.limits['max_output_tokens']:\n                raise ValueError('run reservation budget exhausted')\n",
        ("", ""),
        "plafond local de requetes et de tokens reserves",
    ),
    (
        "V06",
        "scripts/vf_agents.py",
        "    if not args.execute:\n",
        ("", ""),
        "dry-run par defaut sans reseau",
        "    if False:\n",
    ),
    (
        "C01",
        "scripts/hardware_canary.py",
        '        if not 0 <= args.device < 32:\n            raise ValueError("invalid device ordinal")\n',
        ("", ""),
        "ordinal de peripherique borne avant tout acces CUDA",
    ),
    (
        "C02",
        "scripts/hardware_canary.py",
        "            resident_dir.mkdir(mode=0o700)\n",
        ("", ""),
        "repertoire d artefacts residents prive",
        "            resident_dir.mkdir()\n",
    ),
    (
        "C03",
        "scripts/hardware_canary.py",
        '        if previous.get("execution_id") != self.execution_id:\n            raise RuntimeError("H1 receipt ownership changed; refuse replacement")\n',
        ("", ""),
        "propriete du recu H1: pas de remplacement",
    ),
    (
        "C04",
        "scripts/hardware_canary.py",
        '        or sample.get("source", {}).get("revision")\n        != "1355db6a052410cfd62085d94b58866fd0f2c3c5"\n',
        ("", ""),
        "revision epinglee du checkpoint reel",
    ),
    (
        "C05",
        "scripts/hardware_canary.py",
        "                    may_have_executed=bool(self.pending),\n",
        ("", ""),
        "le recu admet qu un noyau a peut-etre tourne",
        "                    may_have_executed=False,\n",
    ),
    (
        "C06",
        "scripts/hardware_canary.py",
        '        if not args.ack_hardware_gate:\n            raise PermissionError(\n                "--ack-hardware-gate is mandatory before any GPU action"\n            )\n',
        ("", ""),
        "acquittement explicite de la porte materielle",
    ),
    (
        "D01",
        "scripts/h1_preflight.py",
        '        if actual != expected:\n            raise PreflightError(f"SHA-256 mismatch for {rel}")\n',
        ("", ""),
        "le destinataire detecte un octet altere dans le paquet",
    ),
    (
        "D02",
        "scripts/h1_preflight.py",
        '        if cur.is_symlink():\n            raise PreflightError(f"symlink is forbidden in bundle path: {rel}")\n',
        ("", ""),
        "aucun lien symbolique dans un chemin du paquet",
    ),
    (
        "D03",
        "scripts/h1_preflight.py",
        '        if rel in result:\n            raise PreflightError(f"duplicate {path.name} path: {rel}")\n',
        ("", ""),
        "pas de chemin duplique dans un manifeste",
    ),
    (
        "D04",
        "scripts/h1_preflight.py",
        '    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):\n        raise PreflightError(f"unsafe manifest path: {value!r}")\n',
        ("", ""),
        "chemin de manifeste sans traversee",
    ),
    (
        "D05",
        "scripts/h1_preflight.py",
        '    if path.as_posix() != value:\n        raise PreflightError(f"manifest path is not canonical POSIX form: {value!r}")\n',
        ("", ""),
        "forme POSIX canonique exigee",
    ),
    (
        "D06",
        "scripts/h1_preflight.py",
        '    if not path.is_file():\n        raise PreflightError(f"bundle path is not a regular file: {rel}")\n',
        ("", ""),
        "chaque entree du paquet est un fichier regulier",
    ),
    (
        "E01",
        "scripts/package_h1.py",
        '        or cpu_proof.get("sources_unchanged") is not True\n',
        ("", ""),
        "le recu CPU doit certifier des sources inchangees",
    ),
    (
        "E02",
        "scripts/package_h1.py",
        '        or any(step.get("exit_code") != 0 for step in cpu_proof["steps"])\n',
        ("", ""),
        "aucune etape du gate en echec dans le recu emballe",
    ),
    (
        "E03",
        "scripts/package_h1.py",
        '    if digest(args.binary) != cpu_proof.get(\n        "binary_sha256"\n    ) or args.binary.stat().st_size != cpu_proof.get("binary_bytes"):\n        raise ValueError("CPU binary differs from qualification receipt")\n',
        ("", ""),
        "le binaire emballe est celui qualifie",
    ),
    (
        "E04",
        "scripts/package_h1.py",
        '        if digest(candidate) != expected:\n            raise ValueError(f"source differs from CPU qualification receipt: {name}")\n',
        ("", ""),
        "chaque source emballee correspond au recu CPU",
    ),
    (
        "E05",
        "scripts/package_h1.py",
        '    if out.exists():\n        raise FileExistsError("inspect existing bundle before any new packaging")\n',
        ("", ""),
        "ne jamais ecraser un paquet existant",
    ),
    (
        "E06",
        "scripts/package_h1.py",
        '    if args.cuda_license.name != "LICENSE" or not args.cuda_license.is_file():\n        raise ValueError("provide the unchanged CUDA runtime component license")\n',
        ("", ""),
        "la licence du composant CUDA voyage avec les octets redistribues",
    ),
    (
        "F01",
        "adapters/skinny_moe.py",
        '        if torch.unique(owned).numel() != owned.numel():\n            raise ValueError("distinct global experts alias one physical stack")\n',
        ("", ""),
        "deux experts globaux ne partagent pas une pile physique",
    ),
    (
        "F02",
        "adapters/skinny_moe.py",
        '        raise ValueError("rank dispatch has stale or incomplete Stamp")\n',
        ("", ""),
        "identite complete run/modele/plan/couche/batch/epoch",
        "        pass\n",
    ),
    (
        "F03",
        "adapters/skinny_moe.py",
        '            or resident[physical] != s["local"]\n',
        ("", ""),
        "doublon de slot ou resident physique etranger",
    ),
    (
        "F04",
        "adapters/skinny_moe.py",
        "            or not bool(coefficient == topk_weights[token, slot])\n",
        ("", ""),
        "les coefficients restent identiques: aucun renormalisage cache",
    ),
    (
        "F05",
        "adapters/skinny_moe.py",
        '    if getattr(layer, "apply_router_weight_on_input", False):\n        raise ValueError("Skinny applies router weights on output")\n',
        ("", ""),
        "Skinny applique les poids du routeur en sortie",
    ),
    (
        "F06",
        "adapters/skinny_moe.py",
        '    if not isinstance(resident, list) or len(resident) != e or len(set(resident)) != e:\n        raise ValueError("physical resident order differs from stack")\n',
        ("", ""),
        "ordre physique resident identique a la pile",
    ),
    (
        "G01",
        "scripts/crosscheck.py",
        "            assert math.isfinite(x) and math.isclose(x, y, rel_tol=2e-11, abs_tol=2e-11), (x, y)\n",
        ("", ""),
        "la tolerance reellement appliquee par l oracle",
        "            assert math.isfinite(x) and math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-4), (x, y)\n",
    ),
    (
        "G02",
        "scripts/crosscheck.py",
        "    detected=False\n    try:close([[1.0]],[[1.001]])\n    except AssertionError:detected=True\n    assert detected\n",
        ("", ""),
        "le controle de sensibilite interne de l oracle",
    ),
    (
        "G03",
        "scripts/planner_crosscheck.py",
        "ATOL = 2e-9  # Decimal transfer costs and f64 roundoff; never a GPU/quantization tolerance.\n",
        ("", ""),
        "la tolerance de l oracle de placement",
        "ATOL = 1e-2  # Decimal transfer costs and f64 roundoff; never a GPU/quantization tolerance.\n",
    ),
    (
        "H01",
        "scripts/checkpoint_crosscheck.py",
        "        torch.testing.assert_close(actual, expected, rtol=0, atol=0)\n",
        ("", ""),
        "le codec checkpoint est compare au bit",
        "        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)\n",
    ),
    (
        "H02",
        "scripts/checkpoint_crosscheck.py",
        '        "atol": 0.0,\n',
        ("", ""),
        "le recu annonce la tolerance reellement appliquee",
        '        "atol": 1e-6,\n',
    ),
    (
        "H03",
        "scripts/resident_crosscheck.py",
        "            assert actual[key] == expected[key], (label, key)\n",
        ("", ""),
        "l oracle resident compare exactement",
        "            assert abs(float(actual[key]) - float(expected[key])) < 1e-6, (\n                label,\n                key,\n            )\n",
    ),
    (
        "K1",
        "src/launch.rs",
        "    let blocks = counts.iter().map(|&n| n.div_ceil(8)).sum::<usize>();\n",
        ("", ""),
        "nombre de blocs du lancement",
        "    let blocks = counts.iter().map(|&n| n.div_ceil(4)).sum::<usize>();\n",
    ),
    (
        "K2",
        "src/moe.rs",
        "        f64::from(1024 + fraction) * 2.0f64.powi(i32::from(exponent) - 25)\n",
        ("completion", "exact_half_extremes_and_signed_zero"),
        "decodage FP16 des sorties compactes",
        "        f64::from(1024 + fraction) * 2.0f64.powi(i32::from(exponent) - 24)\n",
    ),
    (
        "K3",
        "src/resident.rs",
        "            exact_f32(expert.down.global_dequant)?,\n",
        ("", ""),
        "facteur global du plan down dans la pile residente",
        "            exact_f32(expert.down.global_dequant * 2.0)?,\n",
    ),
    (
        "K4",
        "src/quant.rs",
        "        if max_scale * gm >= 65_520.0 {\n",
        ("", ""),
        "seuil binary16 du domaine d echelle SM70",
        "        if max_scale * gm >= 1e30 {\n",
    ),
    (
        "K5",
        "src/main.rs",
        '                "payload_bytes":stack.payload_bytes(),\n',
        ("", ""),
        "taille de charge annoncee par l inspection",
        '                "payload_bytes":stack.payload_bytes() + 1,\n',
    ),
    (
        "K6",
        "src/topology.rs",
        "                let cost = l.latency_us + (bytes as f64 / l.bandwidth_bytes_per_s) * 1e6;\n",
        ("", ""),
        "cout de transfert du modele analytique",
        "                let cost = l.latency_us + (bytes as f64 / l.bandwidth_bytes_per_s) * 2e6;\n",
    ),
    (
        "K7",
        "src/planner.rs",
        "    *denominator += BigUint::from(weight) << (weight_exponent + 1074) as usize;\n",
        ("", ""),
        "agregation exacte du score pondere",
        "    *denominator += BigUint::from(weight) << (weight_exponent + 1073) as usize;\n",
    ),
    (
        "K8",
        "src/pipeline.rs",
        "                let candidate = before.max(cost);\n",
        ("", ""),
        "goulot du pipeline: maximum et non somme",
        "                let candidate = before + cost;\n",
    ),
    # A39: the caps that make Q07 equivalent; each is widened, not removed.
    (
        "M1",
        "src/resident.rs",
        "        || p.experts.len() > 1024\n",
        ("", ""),
        "plafond de 1024 experts physiques",
        "        || p.experts.len() > 4096\n",
    ),
    (
        "M2",
        "src/resident.rs",
        "        || p.resident_order.len() > 1024\n",
        ("", ""),
        "plafond de 1024 entrees d'ordre resident",
        "        || p.resident_order.len() > 4096\n",
    ),
    (
        "M3",
        "src/resident.rs",
        "            || d.slots.len() > 262144\n",
        ("", ""),
        "plafond de 262144 slots par dispatch",
    ),
]


def compilable(rel, text):
    """A Python mutant must parse; a Rust mutant is judged by cargo's own output."""
    if rel.endswith(".py"):
        try:
            ast.parse(text)
        except SyntaxError:
            return False
    return True


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


NEEDED_FROM_SKIPPED = ("evidence/continuation_a23/CUDA_COMPILED.json",)


def copy_source(source, dest):
    def ignore(directory, names):
        return [n for n in names if n in SKIP]

    shutil.copytree(source, dest, ignore=ignore, symlinks=False)
    # Some canonical tests legitimately read a receipt under evidence/, which the
    # ignore list drops. Without these the baseline itself fails and every mutant
    # that reaches that step looks killed for the wrong reason.
    for rel in NEEDED_FROM_SKIPPED:
        origin = source / rel
        if origin.is_file():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)


COSMETIC_GATE_STEPS = ("rustfmt", "clippy", "python_format", "python_lint")


def gate_outcome(failing_steps, environment_errors):
    """One mutant's verdict from its CPU gate run. verify.py stops at its first failing
    step, and the cosmetic steps run before every test step: a gate stopped there ran no
    test, oracle or build, so the mutant is untested, neither killed nor surviving."""
    semantic = [
        s
        for s in failing_steps
        if s not in COSMETIC_GATE_STEPS and s not in environment_errors
    ]
    if semantic:
        return "killed"
    if failing_steps and all(s in COSMETIC_GATE_STEPS for s in failing_steps):
        return "untested"
    return "survived"


def logs_compiled(texts):
    """False when any cargo log shows a compilation error."""
    return not any(
        "error[E" in text or "error: could not compile" in text for text in texts
    )


def failed_tests(texts):
    """Names of the failed tests in cargo or unittest output. A cargo name may hold spaces
    (doctests: "src/x.rs - item (line N)"); unittest prints FAIL:/ERROR: headers."""
    names = set()
    for text in texts:
        names.update(re.findall(r"^test (.+?) \.\.\. FAILED$", text, re.M))
        names.update(re.findall(r"^(?:FAIL|ERROR): (\S+)", text, re.M))
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Base directory; each mutant gets its own subdirectory, removed after its run",
    )
    ap.add_argument(
        "--keep-target",
        action="store_true",
        help="Keep each mutant target directory instead of removing it",
    )
    ap.add_argument("--only", default="", help="comma-separated mutant ids")
    ap.add_argument(
        "--python-tests",
        metavar="SUITE",
        help="Screen Python-only mutations by running this unittest directory "
        "instead of the gate; requires --vf-binary",
    )
    ap.add_argument(
        "--vf-binary",
        type=Path,
        help="Prebuilt vf for --python-tests; the mutation is in Python so the "
        "canonical binary is the right input",
    )
    ap.add_argument(
        "--control",
        action="store_true",
        help="First run the same command on an unmutated copy; abort unless it passes",
    )
    ap.add_argument(
        "--gate-python",
        type=Path,
        help="Run the whole CPU gate with this interpreter instead of cargo test, so the "
        "independent Python oracles also get a chance to notice the mutation",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="run the whole suite instead of the per-case selector",
    )
    args = ap.parse_args()
    wanted = {x for x in args.only.split(",") if x}
    source = args.source.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # One target directory per mutant. A shared one lets cargo consider a test
    # binary built from another tree fresh, which reports a mutant as surviving
    # when the stale binary never contained the mutation at all.
    target_base = args.target_dir.resolve()
    target_base.mkdir(parents=True, exist_ok=True)
    results = []
    if args.control:
        directory = out / "CONTROL"
        copy_source(source, directory)
        target = target_base / "CONTROL"
        env["CARGO_TARGET_DIR"] = str(target)
        if args.python_tests:
            argv = [
                str(args.gate_python),
                "-m",
                "unittest",
                "discover",
                "-s",
                args.python_tests,
            ]
            env["VF_BINARY"] = str(args.vf_binary.resolve(strict=True))
            env["PYTHONPATH"] = str(directory)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
        elif args.gate_python:
            argv = [
                str(args.gate_python),
                "scripts/verify.py",
                "--run-id",
                "control",
            ]
        else:
            argv = ["cargo", "test", "--locked", "--all-targets"]
        run = subprocess.run(
            argv,
            cwd=directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600,
        )
        (directory / "LOG.txt").write_text(run.stdout)
        if not args.keep_target:
            shutil.rmtree(target, ignore_errors=True)
        print(f"CONTROL unmutated copy exit={run.returncode}", flush=True)
        if run.returncode != 0:
            (out / "SUMMARY.json").write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        status="CONTROL_FAILED",
                        detail="The unmutated copy does not pass; no mutant verdict is valid.",
                        argv=argv,
                        cwd=str(directory),
                        log="CONTROL/LOG.txt",
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(dict(status="CONTROL_FAILED")))
            return 1
    for case in CASES:
        name, rel, clause, (test_binary, selector), invariant = case[:5]
        if wanted and name not in wanted:
            continue
        replacement = case[5] if len(case) > 5 else ""
        base = (source / rel).read_text()
        assert base.count(clause) == 1, (name, "anchor count", base.count(clause))
        mutant_text = base.replace(clause, replacement)
        if not compilable(rel, mutant_text):
            raise AssertionError(f"{name}: mutated {rel} does not parse")
        directory = out / name
        copy_source(source, directory)
        (directory / rel).write_text(mutant_text)
        if args.python_tests:
            argv = [
                str(args.gate_python),
                "-m",
                "unittest",
                "discover",
                "-s",
                args.python_tests,
            ]
            env["VF_BINARY"] = str(args.vf_binary.resolve(strict=True))
            env["PYTHONPATH"] = str(directory)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
        elif args.gate_python:
            argv = [
                str(args.gate_python),
                "scripts/verify.py",
                "--run-id",
                f"mutant-{name.lower()}",
            ]
        elif args.full:
            argv = ["cargo", "test", "--locked", "--all-targets"]
        else:
            argv = ["cargo", "test", "--locked", "--test", test_binary]
            if selector:
                argv.append(selector)
        target = target_base / name
        env["CARGO_TARGET_DIR"] = str(target)
        run = subprocess.run(
            argv,
            cwd=directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
        )
        (directory / "LOG.txt").write_text(run.stdout)
        # Where the tests report: the run's own output, or verify.py's per-step logs.
        outputs = [run.stdout]
        failing_steps = []
        gate_status = None
        untested = False
        if args.gate_python:
            summary = (
                directory / "evidence/runs" / f"mutant-{name.lower()}" / "summary.json"
            )
            if summary.is_file():
                report = json.loads(summary.read_text())
                gate_status = report.get("status")
                failing_steps = [
                    step["step"] for step in report["steps"] if step["exit_code"] != 0
                ]
                # verify.py writes cargo's output to per-step logs, not its stdout.
                outputs = [
                    (summary.parent / step["log"]).read_text()
                    for step in report["steps"]
                    if (summary.parent / step["log"]).is_file()
                ]
        compiled = logs_compiled(outputs)
        totals = [
            pair
            for text in outputs
            for pair in re.findall(
                r"test result: \w+\. (\d+) passed; (\d+) failed", text
            )
        ]
        passed = sum(int(a) for a, _ in totals) if totals else None
        failed = sum(int(b) for _, b in totals) if totals else None
        if args.python_tests:
            # unittest does not print cargo's "test result:" line, and the control
            # run already proved the unmutated suite passes, so a non-zero exit
            # here is the mutation unless an import failed.
            imported = (
                "ModuleNotFoundError" not in run.stdout
                and "Start directory is not importable" not in run.stdout
            )
            environment_errors = [] if imported else ["discovery"]
            killed = bool(compiled and imported and run.returncode != 0)
        elif args.gate_python:
            # A rustfmt or clippy failure is a cosmetic artifact of the edit, not a
            # semantic kill, so it is reported separately rather than counted. A step
            # that failed on a missing import is an environment error, and an
            # environment error is not a passing test.
            # Formatting and lint steps fail because the edit changed layout, not
            # behaviour; they are reported but never counted as a semantic kill.
            cosmetic = ("rustfmt", "clippy", "python_format", "python_lint")
            semantic = [x for x in failing_steps if x not in cosmetic]
            env_errors = []
            for step in semantic:
                log = (
                    directory
                    / "evidence/runs"
                    / f"mutant-{name.lower()}"
                    / f"{step}.log"
                )
                if log.is_file() and (
                    "ModuleNotFoundError" in log.read_text()
                    or "ImportError" in log.read_text()
                ):
                    env_errors.append(step)
            semantic = [x for x in semantic if x not in env_errors]
            environment_errors = env_errors
            outcome = gate_outcome(failing_steps, env_errors)
            untested = outcome == "untested"
            killed = bool(compiled and run.returncode != 0 and outcome == "killed")
        else:
            environment_errors = []
            killed = bool(compiled and run.returncode != 0 and failed)
        failing = failed_tests(outputs)
        results.append(
            dict(
                mutant=name,
                isolated_target_dir=str(target),
                file=rel,
                invariant=invariant,
                clause_removed=clause.strip(),
                replacement=replacement.strip() or None,
                canonical_file_sha256=digest(base),
                mutated_file_sha256=digest(mutant_text),
                compiled=compiled,
                argv=argv,
                cwd=str(directory),
                exit_code=run.returncode,
                tests_passed=passed,
                tests_failed=failed,
                failing_tests=failing,
                gate_status=gate_status,
                failing_gate_steps=failing_steps,
                environment_error_steps=environment_errors,
                untested=untested,
                killed=killed,
                log=f"{name}/LOG.txt",
            )
        )
        print(
            f"{name} {rel} compiled={compiled} exit={run.returncode} "
            f"failed={failed} steps={failing_steps} env_errors={environment_errors} "
            f"untested={untested} killed={killed}",
            flush=True,
        )
        if not args.keep_target:
            shutil.rmtree(target, ignore_errors=True)
    survivors = [r["mutant"] for r in results if not r["killed"] and not r["untested"]]
    untested = [r["mutant"] for r in results if r["untested"]]
    report = dict(
        schema_version=1,
        status="SURVIVORS" if survivors else ("UNTESTED" if untested else "PASS"),
        domain="bounded Rust correctness mutants",
        mutants=results,
        survivors=survivors,
        untested=untested,
        cuda_executed=False,
    )
    (out / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                killed=len(results) - len(survivors) - len(untested),
                survivors=survivors,
                untested=untested,
            )
        )
    )
    return 0 if not survivors and not untested else 1


if __name__ == "__main__":
    raise SystemExit(main())
