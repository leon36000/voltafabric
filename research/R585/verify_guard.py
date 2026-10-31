"""Exercise CLI admission with inert paths and a syntactically shaped test key."""
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
binary = root / '.runs/vf-r576-build/debug/vf-sc117-prefill-greedy'
with tempfile.TemporaryDirectory(dir=root / '.runs/vf-r576-tmp', prefix='r585-') as directory:
    key = Path(directory) / 'known_hosts'
    key.write_text('test-host ssh-ed25519 AAAA\n')
    env = {'PATH': '/usr/bin:/bin', 'VF_LOCAL_WORKER_SHA256': 'a'*64,
           'VF_RANK3_SSH_HOST': 'test-host', 'VF_RANK3_SSH_WORKER': '/worker',
           'VF_RANK3_SSH_WORKER_SHA256': 'b'*64,
           'VF_RANK3_SSH_KNOWN_HOSTS': str(key),
           'VF_RANK3_SSH_KNOWN_HOSTS_SHA256': hashlib.sha256(key.read_bytes()).hexdigest()}
    args = [str(binary), '--plan-only'] + ['/not-present'] * 4 + ['/worker'] + ['/not-present'] * 3 + ['48', '1']
    for sticky, expected in [(False, 'SSH workers require sticky expert rank affinity'),
                             (True, 'explicit shared lease identity required for SSH')]:
        candidate = dict(env)
        if sticky:
            candidate['VF_EXPERT_RANK_AFFINITY'] = 'sticky'
        run = subprocess.run(args, env=candidate, capture_output=True, text=True, timeout=10)
        assert run.returncode != 0 and expected in run.stderr, (run.returncode, run.stderr)
        print('sticky=', sticky, 'refused=', expected)
