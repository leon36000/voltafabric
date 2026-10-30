#!/usr/bin/env python3
"""Independent IQ2_S gate/up decode with local MIT-licensed codebook."""
import ctypes
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
pack, prefix, grid_path = map(Path, sys.argv[1:])
with pack.open('rb') as f:
    f.seek(2390092800)
    blob = f.read(1971200)
if len(blob)!=1971200 or hashlib.sha256(blob).hexdigest()!='8a17e1a2a5f9a21e11d2bae64fc99a656c810e8ed16d15bed9a09800fccd9cb5':
    raise ValueError('expert differs from prior observation')
grid_raw = grid_path.read_bytes()
if hashlib.sha256(grid_raw).hexdigest()!='f8bceedc04eaa56c338de3dfd8edc88fa9af857da02bbb217b4f996b56ea9996':
    raise ValueError('codebook identity')
grid_list=json.loads(grid_raw)
if len(grid_list)!=1024: raise ValueError('codebook geometry')
grid=[int(v,16).to_bytes(8,'little') for v in grid_list]
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def q8k(values):
    packed=bytearray(); blocks=[]
    for i in range(0,len(values),256):
        block=values[i:i+256]; maxv=0.0; amax=0.0
        for v in block:
            if abs(v)>amax: amax,maxv=abs(v),v
        if amax==0:
            packed+=bytes(292); blocks.append((0.0,[0]*256)); continue
        inv=f32(-127.0/maxv); d=f32(1.0/inv)
        q=[min(127,round(f32(inv*v))) for v in block]
        if any(n < -128 or n > 127 for n in q): raise ValueError('Q8_K code')
        packed+=struct.pack('<f256b16h',d,*q,*[sum(q[j:j+16]) for j in range(0,256,16)])
        blocks.append((d,q))
    return bytes(packed),blocks
libm = ctypes.CDLL('libm.so.6')
libm.expf.argtypes = [ctypes.c_float]
libm.expf.restype = ctypes.c_float
levels = (-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113)
def pack_mid(gate, up):
    mid=[]
    for g,u in zip(gate,up,strict=True):
        den=f32(1.0 + float(libm.expf(f32(-g))))
        mid.append(f32(f32(g/den)*u))
    result=bytearray()
    for start in range(0,640,32):
        group=mid[start:start+32]
        d=f32(max(map(abs,group))/127.0)
        inv=f32(1.0/d) if d else 0.0
        result+=struct.pack('<e32b',d,*[round(f32(v*inv)) for v in group])
    return bytes(result)
results={}
for mode in ('smooth','boundary'):
    inp=Path(str(prefix)+'-'+mode+'-input.f32').read_bytes()
    packed=Path(str(prefix)+'-'+mode+'-quant.bin').read_bytes()
    if len(inp)!=10240 or len(packed)!=2920: raise ValueError('activation geometry')
    values=struct.unpack('<2560f',inp)
    if any(not math.isfinite(v) for v in values): raise ValueError('nonfinite')
    expected,activations=q8k(values)
    if expected!=packed: raise ValueError('Q8_K packing differs')
    reference_matrices={}
    vector_matrices={}
    for matrix,base in (('gate',0),('up',524800)):
        out=Path(str(prefix)+'-'+mode+'-'+matrix+'.f32').read_bytes()
        if len(out)!=2560: raise ValueError('projection geometry')
        got=struct.unpack('<640f',out)
        max_abs=0.0; violations=0; reference=[]
        for row in range(640):
            total=0.0
            for block in range(10):
                start=base+row*820+block*82
                d=struct.unpack_from('<e',blob,start)[0]
                qs=blob[start+2:start+34]; signs=blob[start+34:start+66]
                hi=blob[start+66:start+74]; scales=blob[start+74:start+82]
                ad,aq=activations[block]
                for group in range(8):
                    for sub in range(4):
                        index=qs[group*4+sub] | ((hi[group] << (8-2*sub)) & 0x300)
                        pattern=grid[index]; sign=signs[group*4+sub]
                        scale=(scales[group]&15) if sub<2 else (scales[group]>>4)
                        aqpos=group*32+sub*8
                        dot=sum(pattern[j]*(-1 if sign & (1<<j) else 1)*aq[aqpos+j] for j in range(8))
                        total += float(d)*(scale+0.5)*0.25*float(ad)*dot
            if not math.isfinite(total) or not math.isfinite(got[row]): raise ValueError('nonfinite result')
            reference.append(f32(total))
            error=abs(got[row]-total); max_abs=max(max_abs,error)
            violations += error > 1e-6 + 2e-5*abs(total)
        results[mode+'_'+matrix]={'rows':640,'max_abs':max_abs,'violations':violations,'output_sha256':hashlib.sha256(out).hexdigest()}
        if violations: raise ValueError(mode+' '+matrix+' independent IQ2_S tolerance')
        reference_matrices[matrix]=reference
        vector_matrices[matrix]=got
    scalar_qmid=pack_mid(reference_matrices['gate'],reference_matrices['up'])
    vector_qmid=pack_mid(vector_matrices['gate'],vector_matrices['up'])
    actual_qmid=Path(str(prefix)+'-'+mode+'-midquant.bin').read_bytes()
    intermediate=Path(str(prefix)+'-'+mode+'-intermediate.f32').read_bytes()
    expert_raw=Path(str(prefix)+'-'+mode+'-expert.f32').read_bytes()
    staged_raw=Path(str(prefix)+'-'+mode+'-staged.f32').read_bytes()
    if len(expert_raw)!=10240 or len(staged_raw)!=10240 or len(actual_qmid)!=680 or len(intermediate)!=2560:
        raise ValueError('expert output geometry')
    expert=struct.unpack('<2560f',expert_raw)
    def compare_down(packed):
        mid_blocks=[struct.unpack_from('<e32b',packed,i*34) for i in range(20)]
        max_error=0.0; bad=0; bound_bad=0; max_bound_ratio=0.0; failing_rows=[]
        eps=2.0**-24
        gamma=(64*eps)/(1-64*eps)
        for row in range(2560):
            total=0.0; sum_abs=0.0
            for block in range(20):
                at=1049600+row*360+block*18
                dw=struct.unpack_from('<e',blob,at)[0]
                codes=blob[at+2:at+18]
                am=mid_blocks[block]
                dot=sum(levels[codes[j]&15]*am[1+j]+levels[codes[j]>>4]*am[17+j] for j in range(16))
                term=float(dw)*float(am[0])*dot
                total+=term; sum_abs+=abs(term)
            if not math.isfinite(total) or not math.isfinite(expert[row]): raise ValueError('nonfinite expert')
            diff=abs(expert[row]-total); max_error=max(max_error,diff)
            threshold=1e-6+2e-5*abs(total)
            bound=gamma*sum_abs+1e-37
            max_bound_ratio=max(max_bound_ratio,diff/bound)
            bound_bad+=diff>bound
            if diff>threshold:
                bad+=1
                if len(failing_rows)<8:
                    failing_rows.append({'row':row,'reference':total,'observed':expert[row],
                                         'absolute_error':diff,'sum_abs_terms':sum_abs,'rounding_bound':bound})
        return {'max_abs':max_error, 'violations':bad,'rounding_bound_violations':bound_bad,
                'max_error_over_rounding_bound':max_bound_ratio,'failing_rows':failing_rows}
    results[mode+'_full_expert']={'rows':2560, **compare_down(scalar_qmid),
        'scalar_vs_vector_mid_quant_bytes':sum(a!=b for a,b in zip(scalar_qmid,vector_qmid,strict=True)),
        'scalar_vs_probed_mid_quant_bytes':sum(a!=b for a,b in zip(scalar_qmid,actual_qmid,strict=True)),
        'staged_matches_provider':staged_raw==expert_raw,
        'independent_down_with_probed_quant':compare_down(actual_qmid),
        'intermediate_sha256':hashlib.sha256(intermediate).hexdigest(),
        'output_sha256':hashlib.sha256(expert_raw).hexdigest()}
    results[mode+'_input_sha256']=hashlib.sha256(inp).hexdigest()
    results[mode+'_q8k_sha256']=hashlib.sha256(packed).hexdigest()
print(json.dumps({'scope':'synthetic inputs; independent full expert with IQ2_S codebook, Q8_K, SiLU, Q8_0 and IQ4_NL',
                  'blob_sha256':hashlib.sha256(blob).hexdigest(),'codebook_sha256':hashlib.sha256(grid_raw).hexdigest(),
                  'tolerance':'1e-6 + 2e-5 * abs(reference)','results':results},sort_keys=True))
