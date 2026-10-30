import json,pathlib
from extract_layer import extract
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r299-extracted-second-r1';root.mkdir();source=b/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00002-of-00003.safetensors';sha='6dea9c759a0f941cf9e1cc1501216b0e107966a66c90425050e212efbd053f02'
for layer in range(32,37):
 ids=list(range(256));r=extract(source,sha,root/f'layer{layer}.safetensors',ids,layer);(root/f'layer{layer}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'PASS','layer':layer,'experts':len(ids),'source_sha256':r['source_sha256'],'selected_sha256':r['selected_sha256'],'bytes':r['selected_bytes']}),flush=True)
