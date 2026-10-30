"""Recheck the retained R521 receipt without running an engine or replaying claims."""
import hashlib, json, pathlib, subprocess
from adapters.strata_tokenizer.adapter import load_sc117
from adapters.strata_tokenizer.prompt_runner import run_prompt
root = pathlib.Path('/home/pc1/NAOS/vf-r521-work')
manifest = json.loads((root/'adapters/strata_tokenizer/observed-sc117.json').read_text())
names = ('vocab.json', 'merges.txt', 'token_type.json', 'tokenizer.json')
tokenizer, eos = load_sc117('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s/tokenizer', {k:manifest[k] for k in names})
raw = (root/'result.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'b9b8d0436688ad5695ede0e5783dcf5a55c23d3085cc29ee287ec8222ca7dae2'
receipt = json.loads(raw)['receipt']
def execute():
    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps(receipt).encode(), b'')
    return run_prompt('Hi', tokenizer, eos, '/engine', ['/input']*8, max_new_tokens=1,
                      chat=True, runner=runner)
assert execute()['output']['text'] == 'Hello'
original = receipt['prompt_tokens']
for value in (original[:-1], None, original+[0]):
    receipt['prompt_tokens'] = value
    try: execute()
    except ValueError as exc: assert 'receipt differs from prompt tokens' in str(exc)
    else: raise AssertionError('bad prompt accepted')
del receipt['prompt_tokens']
try: execute()
except ValueError as exc: assert 'receipt differs from prompt tokens' in str(exc)
else: raise AssertionError('missing prompt accepted')
print('PASS: retained R521 receipt accepted; altered prompt fields refused; no engine run')
