"""Bounded CPU reference for greedy decoding of a ModelOpt NVFP4 K16 head.

This is not a GPU kernel, sampling implementation, or complete model. The caller
must verify the checkpoint shard identity and ensure it remains quiescent. Rows are
read in chunks; no full dense vocabulary matrix is built.
"""
import math
import os
import pathlib
import torch
from safetensors import safe_open

class HeadError(ValueError):
    pass

def greedy_nvfp4_head(path, hidden, *, vocab, width, chunk_rows=512):
    if (not isinstance(hidden,torch.Tensor) or hidden.device.type!='cpu'
        or hidden.dtype!=torch.float16 or hidden.ndim!=1 or hidden.numel()!=width
        or not torch.isfinite(hidden).all() or type(vocab) is not int or not 0<vocab<=1_000_000
        or type(width) is not int or not 0<width<=65536 or width%16
        or type(chunk_rows) is not int or not 0<chunk_rows<=1024):
        raise HeadError('invalid bounded head input')
    path=pathlib.Path(path)
    if path.is_symlink() or not path.is_file(): raise HeadError('regular local checkpoint required')
    before=path.stat();fingerprint=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
    lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
    best_id=None;best_value=-math.inf
    with safe_open(str(path),framework='pt',device='cpu') as f, torch.inference_mode():
        codes=f.get_slice('lm_head.weight');scales=f.get_slice('lm_head.weight_scale')
        global_scale=f.get_tensor('lm_head.weight_scale_2')
        if (tuple(codes.get_shape())!=(vocab,width//2) or tuple(scales.get_shape())!=(vocab,width//16)
            or global_scale.numel()!=1 or not torch.isfinite(global_scale).all() or global_scale.item()<=0):
            raise HeadError('invalid checkpoint geometry or global scale')
        g=global_scale.float();h=hidden.double()
        for start in range(0,vocab,chunk_rows):
            end=min(vocab,start+chunk_rows);c=codes[start:end];s=scales[start:end]
            if c.dtype!=torch.uint8 or s.dtype!=torch.float8_e4m3fn or not torch.isfinite(s.float()).all() or (s.float()<0).any():
                raise HeadError('invalid NVFP4 codes or scales')
            ids=torch.stack((c&15,c>>4),dim=-1).reshape(end-start,width).long()
            matrix=(lut[ids]*s.float().repeat_interleave(16,dim=1)*g).half()
            logits=torch.mv(matrix.double(),h).half()
            if not torch.isfinite(logits).all():raise HeadError('nonfinite logits')
            value,index=logits.max(dim=0);v=float(value);i=start+int(index)
            if v>best_value:best_value,best_id=v,i
    after=path.stat()
    if fingerprint!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise HeadError('checkpoint changed during read')
    return {'token_id':best_id,'logit_fp16':best_value,'vocab':vocab}
