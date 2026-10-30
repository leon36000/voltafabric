"""Verify tokenizer-prepared plan against independently reconstructed R516 plan."""
import json, runpy
from pathlib import Path
root=Path(__file__).parent
runpy.run_path(str(root.parent/'R516/verify_plan.py'))
current=json.loads((root/'PLAN.json').read_text())
prior=json.loads((root.parent/'R516/PLAN.json').read_text())
assert set(current)=={'plan_id','prompt_tokens_sha256','prompt_token_ids'}
assert current['plan_id']==prior['plan_id'] and current['prompt_tokens_sha256']==prior['prompt_tokens_sha256']
assert current['prompt_token_ids']==[248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
print('PASS: tokenizer IDs and independent plan match')
