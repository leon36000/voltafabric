"""Independent live-process check of the default post-exit pipe guard."""
import os,sys,time
from adapters.strata_tokenizer.prompt_runner import _run_bounded
if not hasattr(os,'fork'): raise SystemExit('POSIX fork required')
child="import os,time; p=os.fork(); time.sleep(4) if p==0 else os.write(1,b'{\\\"result\\\":\\\"partial\\\"}'); os._exit(0)"
start=time.monotonic()
try:
    _run_bounded([sys.executable,'-c',child])
except RuntimeError as error:
    elapsed=time.monotonic()-start
    assert 'output pipes remain open' in str(error) and 'reconcile' in str(error)
    assert 2.0 <= elapsed < 4.0, elapsed
    print('PASS: direct child exited, inherited pipe refused before descendant exit; no receipt returned')
else:
    raise AssertionError('partial response accepted')
