"""Real Rust worker, with only its new BF16 HELLO capability removed.

This exercises the legacy raw32 protocol, without a fake numeric producer.
The wrapper forwards every body byte unchanged and exits with the worker.
"""

import json
import os
import subprocess
import sys
import threading

worker = subprocess.Popen(sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def incoming():
    try:
        while data := os.read(sys.stdin.fileno(), 65536):
            worker.stdin.write(data)
            worker.stdin.flush()
    except BrokenPipeError:
        pass
    finally:
        worker.stdin.close()


threading.Thread(target=incoming, daemon=True).start()
hello = json.loads(worker.stdout.readline())
assert hello["reply"] == "hello"
hello["hello"]["capabilities"] = [
    cap
    for cap in hello["hello"]["capabilities"]
    if cap != "mxfp4_exec_bf16_exact_raw_v1"
]
sys.stdout.buffer.write(json.dumps(hello).encode() + b"\n")
sys.stdout.buffer.flush()
while data := worker.stdout.read1(65536):
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
sys.exit(worker.wait(timeout=10))
