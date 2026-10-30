#!/usr/bin/env python3
"""Diagnostic pipe proxy: suppress one completed native eviction acknowledgement.

Only the child created by this process is terminated. The marker records the
completed response's identity and hash, never its payload. No retry is made.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading

child = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=None, bufsize=0)
def upstream():
    try:
        while True:
            data=os.read(0,65536)
            if not data: break
            view=memoryview(data)
            while view:
                n=os.write(child.stdin.fileno(),view)
                if n<=0: raise OSError('short proxy write')
                view=view[n:]
    except (BrokenPipeError,OSError):
        pass
    finally:
        try: child.stdin.close()
        except OSError: pass
threading.Thread(target=upstream,daemon=True).start()
rank=None
try:
    while True:
        line=child.stdout.readline(16*1024*1024+2)
        if not line: break
        if len(line)>16*1024*1024+1 or not line.endswith(b'\n'): raise ValueError('invalid bounded reply')
        reply=json.loads(line)
        if reply.get('reply')=='hello': rank=reply['hello']['rank']
        if rank==3 and reply.get('reply')=='native_ggml_evicted':
            marker={'rank':rank,'attempt':reply['attempt'],'stamp':reply['stamp'],
                    'operation_id':reply['operation_id'],'binding':reply['binding'],
                    'bytes':reply['bytes'],'experts':reply['experts'],
                    'response_sha256':hashlib.sha256(line).hexdigest()}
            fd=os.open(sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as f:
                json.dump(marker,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
            break
        sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
finally:
    if child.poll() is None: child.kill()
    child.wait()
