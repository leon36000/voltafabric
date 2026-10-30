"""Check R514's retained receipt against strict prompt validation; never run an engine."""
import hashlib, json, pathlib, subprocess
from adapters.strata_tokenizer.adapter import load_sc117
from adapters.strata_tokenizer.prompt_runner import run_prompt
root = pathlib.Path('/home/pc1/NAOS/vf-r514-work')
manifest = json.loads((root/'adapters/strata_tokenizer/observed-sc117.json').read_text())
names = ('vocab.json', 'merges.txt', 'token_type.json', 'tokenizer.json')
tokenizer, eos = load_sc117('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s/tokenizer', {k:manifest[k] for k in names})
raw = (root/'result.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '9f153c3cc049e309b857715f328407555a0bfec6c2d131aed214470095736d98'
receipt = json.loads(raw)['receipt']
def execute():
    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps(receipt).encode(), b'')
    return run_prompt('Hi', tokenizer, eos, '/engine', ['/input']*8, max_new_tokens=1,
                      chat=True, require_bound_plan=True, runner=runner)
assert execute()['output']['text'] == 'Hello'
original = receipt['plan_id']
for bad in (None, 'r508-'+'a'*64, 'r512-'+'A'*64):
    receipt['plan_id'] = bad
    try: execute()
    except ValueError: pass
    else: raise AssertionError('bad plan accepted')
receipt['plan_id'] = original
receipt.pop('prompt_tokens_sha256')
try: execute()
except ValueError: pass
else: raise AssertionError('missing prompt digest accepted')
print('PASS: real R514 receipt accepted; altered plan and missing prompt digest refused; no engine run')
