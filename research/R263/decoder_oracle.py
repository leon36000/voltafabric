"""Independent CPU composition oracle for the recorded R252 single-layer fixture."""
import hashlib,json,math,os,pathlib
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule
from nvfp4_oracle import expert_forward
torch.set_num_threads(1)
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r263-qwen-full-gate-20261028-r1';model=base/'r188-qwen-checkpoint-pc1-20261028-r1/model';fixture=base/'r262-full-model-fixture-r1/nonexpert.safetensors';source=base/'r262-full-layer-r1/experts.safetensors';receipt=json.loads((root/'receipt.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert receipt['status']=='SUCCESS' and sha(root/'capture.safetensors')==receipt['capture_sha256'] and sha(fixture)=='35c4d00d2dcbc09db0ac89494ceee34820b2ca52d0805b9ac0959418113f4f15'
capture=load_file(str(root/'capture.safetensors'));weights=load_file(str(fixture));pre='model.language_model.layers.0.'
keys=['in_proj_qkv.weight','in_proj_qkv.weight_scale','in_proj_qkv.input_scale','in_proj_z.weight','in_proj_z.weight_scale','in_proj_z.input_scale','out_proj.weight','out_proj.weight_scale','out_proj.input_scale','in_proj_a.weight','in_proj_b.weight','conv1d.weight','A_log','dt_bias','norm.weight']
with safe_open(str(model/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f: w={k:f.get_tensor(pre+'linear_attn.'+k).contiguous() for k in keys}
def fp8(name,x):
 scale=w[name+'.input_scale']; q=(x.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).float(); return (q@w[name+'.weight'].float().T)*scale*w[name+'.weight_scale']
def rms(x,key):
 xf=x.float();return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)*(1+weights[key].float())).half()
def decode(name):
 p=pre+'mlp.shared_expert.'+name+'.'; c=weights[p+'weight'];s=weights[p+'weight_scale'].float();g=weights[p+'weight_scale_2'].float();n,k=c.shape;lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]); ids=torch.stack((c&15,c>>4),-1).reshape(n,k*2).long();return (lut[ids]*s.repeat_interleave(16,1)*g).half().double()
gate,up,down=(decode(k) for k in ('gate_proj','up_proj','down_proj'));shared_gate=weights[pre+'mlp.shared_expert_gate.weight'].half();router=weights[pre+'mlp.gate.weight'].half();x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);history=torch.zeros(1,8192,3);state=None;results=[]
for b in range(2):
 n1=rms(x,pre+'input_layernorm.weight').float().unsqueeze(0)
 qkv=fp8('in_proj_qkv',n1.reshape(1,2048)).reshape(1,1,8192);z=fp8('in_proj_z',n1.reshape(1,2048)).reshape(1,1,32,128);a=F.linear(n1,w['in_proj_a.weight'].float());bb=F.linear(n1,w['in_proj_b.weight'].float())
 h=torch.cat([history,qkv.transpose(1,2)],-1);conv=F.silu(F.conv1d(h,w['conv1d.weight'].float(),groups=8192)).transpose(1,2);history=h[:,:,-3:].clone();q,k,v=conv.split([2048,2048,4096],-1);q=q.reshape(1,1,16,128).repeat_interleave(2,2);k=k.reshape(1,1,16,128).repeat_interleave(2,2);v=v.reshape(1,1,32,128);g=-w['A_log'].float().exp()*F.softplus(a+w['dt_bias'].float());core,state=torch_chunk_gated_delta_rule(q,k,v,g,bb.sigmoid(),chunk_size=4,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True);norm=core*torch.rsqrt(core.square().mean(-1,keepdim=True)+1e-6)*w['norm.weight'].float()*F.silu(z);attn=fp8('out_proj',norm.reshape(1,4096)).reshape(1,2048).half()
 residual=x.float()+attn.float(); n2=rms(residual,pre+'post_attention_layernorm.weight');delta=(n2.float()-capture[f'batch{b}.input'].float()).abs().max().item();assert delta<=0.003,(b,'post_norm',delta)
 logits=F.linear(n2,router);probs=torch.softmax(logits.float(),dim=-1);cw,ids=probs.topk(8,dim=-1);cw=cw/cw.sum(-1,keepdim=True);assert torch.equal(ids,capture[f'batch{b}.ids'])
 acc=torch.zeros(2048,dtype=torch.float64)
 for idx,coef in zip(ids[0].tolist(),cw[0].tolist()):acc+=expert_forward(source,0,idx,n2[0]).double()*float(coef)
 routed=acc.half().unsqueeze(0);gx=(n2.double()@gate.T).half();ux=(n2.double()@up.T).half();inter=(F.silu(gx)*ux).half();shared=(inter.double()@down.T).half()*F.sigmoid(F.linear(n2,shared_gate));combined=(routed+shared).half();output=rms(residual+combined.float(),'model.language_model.norm.weight')
 actual=capture[f'batch{b}.output'];error=(output.float()-actual.float()).abs();maximum=error.max().item();assert maximum<=0.003,(b,'decoder',maximum)
 results.append({'batch':b,'max_post_norm_error':delta,'max_decoder_output_error':maximum,'output_elements':output.numel()})
out={'status':'PASS','cases':results,'atol':0.003,'gpu':False,'scope':'synthetic one-layer CPU fixture, 8 of 256 gate rows; unfused GDN and single-process independent oracle'}
p=base/'r263-decoder-oracle-r1';p.mkdir(mode=0o700);(p/'receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
