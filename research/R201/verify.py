"""Compare captured Rust session output with independent row-major NVFP4 CPU oracle."""
import hashlib,json,math,pathlib,struct,sys
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent/'R199'))
from nvfp4_oracle import expert_forward

def accumulate(routes, rows, forward):
    """Each route's exact coefficient is applied once, in logical slot order."""
    if len(routes)!=len(rows):raise ValueError('token count mismatch')
    result=[]
    for token,slot_list in enumerate(routes):
        if not slot_list or len(slot_list)>64:raise ValueError('invalid route count')
        seen=set();acc=None
        for slot in slot_list:
            if set(slot)!={'expert','weight'} or type(slot['expert'])!=int or slot['expert']<0 or slot['expert'] in seen:
                raise ValueError('invalid or duplicate expert')
            weight=slot['weight']
            if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or weight<0:
                raise ValueError('invalid weight')
            seen.add(slot['expert']);value=forward(slot['expert'],rows[token]).to(torch.float64)
            if value.ndim!=1 or not torch.isfinite(value).all():raise ValueError('invalid expert output')
            if acc is None:acc=torch.zeros_like(value,dtype=torch.float64)
            if acc.shape!=value.shape:raise ValueError('hidden shape mismatch')
            acc=acc+value*float(weight)
        result.append(acc)
    return torch.stack(result)

def _hash(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()

def verify(*,source,source_sha,input_path,input_sha,output_path,output_sha,atol):
    if not math.isfinite(atol) or not 0<atol<=0.01:raise ValueError('invalid predeclared tolerance')
    if any(_hash(path)!=sha for path,sha in [(source,source_sha),(input_path,input_sha),(output_path,output_sha)]):raise ValueError('evidence hash mismatch')
    request=json.loads(pathlib.Path(input_path).read_bytes());reply=json.loads(pathlib.Path(output_path).read_bytes());result=reply['result'];routing=request['routing']
    if not reply.get('ok') or not result.get('ok') or result.get('error') is not None:raise ValueError('session not successful')
    if request.get('inputs_fp16_bits') is None:raise ValueError('raw FP16 inputs required')
    bits=request['inputs_fp16_bits'];rows=[torch.tensor([struct.unpack('<e',struct.pack('<H',x))[0] for x in row],dtype=torch.float16) for row in bits]
    if any(not torch.isfinite(row).all() for row in rows):raise ValueError('nonfinite input')
    layer=routing['stamp']['layer'];expected=accumulate(routing['routes'],rows,lambda expert,row:expert_forward(source,layer,expert,row))
    records=result['recorded'];requested=request['record']
    if len(records)!=len(requested) or [x['batch'] for x in records]!=requested or result['batches_ok']!=request['batches']:
        raise ValueError('missing or stale batch output')
    stats=[]
    for record in records:
        actual=torch.tensor(record['output'],dtype=torch.float64)
        if actual.shape!=expected.shape or not torch.isfinite(actual).all():raise ValueError('invalid output')
        error=(actual-expected).abs();maximum=float(error.max())
        stats.append({'batch':record['batch'],'max_abs_error':maximum,'per_token_max':[float(x) for x in error.max(dim=1).values]})
        if maximum>atol:raise ValueError(f'oracle mismatch batch {record["batch"]}: {maximum} > {atol}')
    return {'status':'PASS','source_sha256':source_sha,'input_sha256':input_sha,'output_sha256':output_sha,'atol':atol,'records':stats,'expert_ids':sorted({s['expert'] for row in routing['routes'] for s in row})}
