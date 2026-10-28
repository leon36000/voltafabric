"""Real CPU session startup interleavings; no GPU, SSH or timing speedup assertions."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "r18_crosscheck", ROOT / "scripts/mxfp4_session_crosscheck.py"
)
cross = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cross)
BINARY = Path(os.environ.get("VF_BINARY", ROOT / "target/release/vf")).resolve()
WORKER = BINARY.parent / "vf-rank-session"


@unittest.skipUnless(
    BINARY.is_file() and WORKER.is_file(), "native CLI and CPU session worker required"
)
class ParallelStartup(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("TMPDIR", ROOT / "evidence/adapter-temp"))
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="r18-startup-", dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def execute(
        self, parallelism=None, fault=None, barrier=False, workers=2, fail_rank="9"
    ):
        payload, expected = cross.fixture(BINARY)
        if workers > 2:
            template = payload["workers"][0]
            raw = cross.ARTIFACT.read_bytes()
            size = int.from_bytes(raw[:8], "little")
            header = json.loads(raw[8 : 8 + size])
            sources, copies, experts, owners, orders = [], [], [], {}, {}
            for index in range(workers):
                rank = index + 2
                source = copy.deepcopy(template)
                source["endpoint"]["rank"] = source["binding"]["rank"] = rank
                renamed = {}
                for key, value in header.items():
                    parts = key.split(".")
                    if key != "__metadata__":
                        parts[4] = str(int(parts[4]) + 1000 * (index + 1))
                    renamed[".".join(parts)] = value
                encoded = json.dumps(renamed, separators=(",", ":")).encode()
                encoded += b" " * ((-len(encoded)) % 8)
                shard = len(encoded).to_bytes(8, "little") + encoded + raw[8 + size :]
                artifact = self.directory / ("weights-" + str(rank))
                artifact.write_bytes(shard)
                source["artifact_path"] = str(artifact)
                source["binding"]["artifact_sha256"] = hashlib.sha256(shard).hexdigest()
                for item in source["binding"]["experts"]:
                    item["expert"] += 1000 * (index + 1)
                    copies.append(
                        dict(expert=item["expert"], rank=rank, local=item["local"])
                    )
                    experts.append(item["expert"])
                chosen = source["binding"]["experts"][0]["expert"]
                owners[str(chosen)] = rank
                orders[str(rank)] = [
                    item["local"] for item in source["binding"]["experts"]
                ]
                sources.append(source)
            payload["workers"] = sources
            for batch in payload["batches"]:
                batch["problem"].update(
                    experts=experts,
                    routes=[
                        [dict(expert=int(expert), weight=1.0)] for expert in owners
                    ],
                    preferred_owners=owners,
                    resident_order=orders,
                )
                batch["problem"]["registry"]["copies"] = copies
                batch["inputs"] = [
                    dict(token=i, value_bits=[0] * 32) for i in range(workers)
                ]
            expected = [[[0] * 32 for _ in range(workers)] for _ in payload["batches"]]
        if parallelism is not None:
            payload["startup_parallelism"] = parallelism
        hello = subprocess.run(
            [str(WORKER)],
            input=b'{"op":"hello","protocol":1,"rank":9}\n{"op":"shutdown"}\n',
            capture_output=True,
            timeout=5,
        )
        self.assertEqual(hello.returncode, 0, hello.stderr)
        line = hello.stdout.splitlines()[0]
        launcher = self.directory / "launcher.py"
        launcher.write_text(
            """import json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]);rank=sys.argv[2];worker=sys.argv[3];fault=sys.argv[4];barrier=sys.argv[5]=='yes';ranks=json.loads(sys.argv[6]);width=int(sys.argv[7]);fail_rank=sys.argv[8]
wave=ranks[(ranks.index(rank)//width)*width:((ranks.index(rank)//width)+1)*width]
(root/(rank+'.pid')).write_text(str(os.getpid()))
if barrier:
 deadline=time.monotonic()+3
 while not all((root/(r+'.pid')).exists() for r in wave):
  if time.monotonic()>deadline:raise SystemExit(88)
  time.sleep(.005)
if fault=='before_hello' and rank==fail_rank:raise SystemExit(66)
if fault=='after_load' and rank==fail_rank:
 request=json.loads(sys.stdin.buffer.readline());assert request['op']=='hello'
 template=json.loads("""
            + repr(line)
            + """);template['hello']['rank']=int(rank);template['hello']['pid']=os.getpid()
 sys.stdout.write(json.dumps(template)+'\\n');sys.stdout.flush()
 header=json.loads(sys.stdin.buffer.readline());remaining=header['artifact_bytes']
 assert 0<remaining<1048576
 while remaining:
  data=sys.stdin.buffer.read(min(65536,remaining))
  if not data:raise SystemExit(67)
  remaining-=len(data)
 (root/'load-consumed').write_text('all compressed bytes received')
 raise SystemExit(69)
os.execv(worker,[worker])
"""
        )
        for worker in payload["workers"]:
            worker["endpoint"]["argv"] = [
                sys.executable,
                "-B",
                str(launcher),
                str(self.directory),
                str(worker["endpoint"]["rank"]),
                str(WORKER),
                fault or "none",
                "yes" if barrier else "no",
                json.dumps([str(w["endpoint"]["rank"]) for w in payload["workers"]]),
                str(parallelism if workers > 2 else 2),
                fail_rank,
            ]
        q = subprocess.run(
            [str(BINARY), "session-run-mxfp4", "-"],
            input=json.dumps(payload).encode(),
            capture_output=True,
            timeout=20,
        )
        result = json.loads(q.stdout)
        for path in self.directory.glob("*.pid"):
            pid = int(path.read_text())
            try:
                cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            except FileNotFoundError:
                continue
            self.assertNotIn(
                str(self.directory).encode(), cmd, "launcher survived result"
            )
            self.assertNotIn(str(WORKER).encode(), cmd, "worker survived result")
        return q, result, expected

    def test_default_serial_control(self):
        q, result, expected = self.execute()
        self.assertEqual(q.returncode, 0, result)
        self.assertEqual(
            [b["output_bits"] for b in result["result"]["batches"]], expected
        )

    def test_parallel_workers_cross_real_startup_barrier(self):
        q, result, expected = self.execute(2, barrier=True)
        self.assertEqual(q.returncode, 0, result)
        report = result["result"]
        self.assertTrue(report["ok"])
        self.assertEqual([b["output_bits"] for b in report["batches"]], expected)
        self.assertEqual(len(report["loads"]), 2)
        self.assertTrue(
            all(
                t[1]["loads_transferred"] == 1 and t[1]["execs"] == 3
                for t in report["data_phase_traffic"]
            )
        )
        self.assertTrue(
            all(c["exit_code"] == 0 and c["bye_execs"] == 3 for c in report["closes"])
        )

    def test_serial_does_not_cross_two_worker_barrier(self):
        q, result, _ = self.execute(barrier=True)
        self.assertEqual(q.returncode, 2)
        self.assertFalse(result["result"]["ok"])
        self.assertEqual(result["result"]["batches"], [])
        self.assertFalse((self.directory / "9.pid").exists())

    def test_parallel_failure_before_hello_closes_other_worker(self):
        q, result, _ = self.execute(2, fault="before_hello", barrier=True)
        self.assertEqual(q.returncode, 2)
        report = result["result"]
        self.assertFalse(report["ok"])
        self.assertEqual(report["batches"], [])
        self.assertEqual([c["rank"] for c in report["closes"]], [2])
        self.assertEqual(report["closes"][0]["exit_code"], 0)
        self.assertEqual(report["closes"][0]["bye_execs"], 0)

    def test_parallel_failure_after_load_keeps_all_close_receipts(self):
        q, result, _ = self.execute(2, fault="after_load", barrier=True)
        self.assertEqual(q.returncode, 2)
        report = result["result"]
        self.assertFalse(report["ok"])
        self.assertEqual(report["batches"], [])
        self.assertTrue((self.directory / "load-consumed").exists())
        self.assertEqual({c["rank"] for c in report["closes"]}, {2, 9})
        self.assertEqual(
            next(c for c in report["closes"] if c["rank"] == 9)["exit_code"], 69
        )
        self.assertEqual(
            next(c for c in report["closes"] if c["rank"] == 2)["bye_execs"], 0
        )

    def test_four_workers_start_together_then_next_wave(self):
        q, result, expected = self.execute(4, workers=5, barrier=True)
        self.assertEqual(q.returncode, 0, result)
        report = result["result"]
        self.assertEqual([b["output_bits"] for b in report["batches"]], expected)
        self.assertEqual(len(report["loads"]), 5)
        self.assertEqual({c["rank"] for c in report["closes"]}, {2, 3, 4, 5, 6})
        self.assertTrue(
            all(c["exit_code"] == 0 and c["bye_execs"] == 3 for c in report["closes"])
        )

    def test_later_wave_load_failure_closes_all_earlier_workers(self):
        q, result, _ = self.execute(
            2, fault="after_load", workers=5, fail_rank="6", barrier=True
        )
        self.assertEqual(q.returncode, 2)
        report = result["result"]
        self.assertEqual(report["batches"], [])
        self.assertTrue((self.directory / "load-consumed").exists())
        self.assertEqual({c["rank"] for c in report["closes"]}, {2, 3, 4, 5, 6})
        self.assertEqual(len(report["loads"]), 4)
        self.assertTrue(
            all(c["bye_execs"] == 0 for c in report["closes"] if c["rank"] != 6)
        )

    def test_first_wave_failure_prevents_later_spawns(self):
        q, result, _ = self.execute(
            2, fault="before_hello", workers=5, fail_rank="3", barrier=True
        )
        self.assertEqual(q.returncode, 2)
        self.assertEqual(result["result"]["batches"], [])
        self.assertEqual({p.stem for p in self.directory.glob("*.pid")}, {"2", "3"})
        self.assertEqual([c["rank"] for c in result["result"]["closes"]], [2])

    def test_invalid_parallelism_never_spawns(self):
        for limit in [0, 5, 64]:
            with self.subTest(limit=limit):
                q, result, _ = self.execute(limit)
                self.assertEqual(q.returncode, 2)
                self.assertFalse(result["ok"])
                self.assertFalse(list(self.directory.glob("*.pid")))


if __name__ == "__main__":
    unittest.main()
