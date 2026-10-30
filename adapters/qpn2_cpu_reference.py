"""Exact-scale CPU QPN2 unpack oracle; not an SM70 or Blackwell kernel.

QPN2's permuted E2M1 codes and unsigned E4M3 group scales are decoded, multiplied
with the original F32 per-tile global scale in float64, and rounded once to FP16.
This deliberately differs from the existing SM70 provider's intermediate FP16
roundings. The caller must authenticate and validate the resident artifact.
"""
import torch

def unpack_exact(codes, scales, global_scales, rows, columns):
    if (type(rows) is not int or type(columns) is not int or not 0<rows<=65536
        or rows%32 or not 64<=columns<=65536 or columns%64
        or not isinstance(codes,torch.Tensor) or not isinstance(scales,torch.Tensor)
        or not isinstance(global_scales,torch.Tensor)
        or any(t.device.type!='cpu' for t in (codes,scales,global_scales))
        or codes.dtype!=torch.uint8 or scales.dtype!=torch.uint8 or global_scales.dtype!=torch.float32
        or codes.numel()!=rows*columns//2 or scales.numel()!=rows*columns//16
        or global_scales.numel()!=rows//32 or not torch.isfinite(global_scales).all()
        or (global_scales<=0).any() or (scales>=127).any()):
        raise ValueError('invalid QPN2 exact CPU input')
    tiles,groups=rows//32,columns//16
    c=codes.reshape(tiles,groups,32,8);s=scales.reshape(tiles,groups,32)
    lane=torch.arange(32);columns_perm=((lane>>2)&3)*8+(lane&3)+((lane&16)>0).long()*4
    inverse_rows=torch.argsort(columns_perm);inverse_k=torch.argsort(torch.tensor([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]))
    nibble=torch.stack((c&15,c>>4),-1).reshape(tiles,groups,32,16)
    nibble=nibble[:,:,inverse_rows,:][:,:,:,inverse_k].permute(0,2,1,3).reshape(rows,columns).long()
    sf=s[:,:,inverse_rows].permute(0,2,1).reshape(rows,groups).to(torch.int32).repeat_interleave(16,dim=1)
    scale=((sf<<7).to(torch.int16).view(torch.float16)*torch.tensor(256.,dtype=torch.float16)).double()
    factor=global_scales.reshape(-1).repeat_interleave(32).reshape(rows,1).double()
    lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
    out=(lut[nibble]*scale*factor).half()
    if not torch.isfinite(out).all():raise ValueError('decoded QPN2 exceeds FP16 range')
    return out
