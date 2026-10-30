"""Check compatibility with a preserved successful v2 receipt, without an engine run."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from adapters.strata_tokenizer.adapter import load_sc117
from adapters.strata_tokenizer.prompt_runner import run_prompt
if len(sys.argv)!=5: raise SystemExit('usage: verify_saved_receipt RESULT PROMPT TOKENIZER MANIFEST')
result,prompt,tok,manifest=map(Path,sys.argv[1:]); raw=result.read_bytes()
if hashlib.sha256(raw).hexdigest()!='6b1b2a6e75c11a843b6a82bd21aea90c5f1183463f2c2a7418268af280dcdeca':raise ValueError('saved receipt digest')
x=json.loads(raw); m=json.loads(manifest.read_text()); hashes={k:m[k] for k in ('vocab.json','merges.txt','token_type.json','tokenizer.json')};tokenizer,eos=load_sc117(tok,hashes)
for key in ('VF_EXPECTED_PLAN_ID','VF_RANK2_SSH_HOST','VF_RANK3_SSH_HOST'):os.environ.pop(key,None)
def fake(args,**kwargs):return subprocess.CompletedProcess(args,0,json.dumps(x['receipt'],separators=(',',':')).encode(),b'')
r=run_prompt(json.loads(prompt.read_text()),tokenizer,eos,'/unused',['/unused']*8,max_new_tokens=20,chat=True,runner=fake)
assert r['output']==x['output'] and r['prompt_token_ids']==x['prompt_token_ids']
print('PASS: retained v2 receipt and prompt produce the same decoded output; no engine run')
