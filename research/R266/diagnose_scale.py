"""Offline R265 per-expert QPN2 vs original NVFP4 arithmetic diagnosis."""
import ast,hashlib,json,pathlib,sys,torch
from safetensors import safe_open
from safetensors.torch import load_file
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';root=B/'.runs/r266-expert-diagnostics-20261028';torch.set_num_threads(1)
cap=root/'capture.safetensors';assert hashlib.sha256(cap.read_bytes()).hexdigest()=='8444bce6a7ecdc76a21463b01e62ddf3c108d18887037ff6a2c6dfb1503adcd5';c=load_file(str(cap));x=c['batch0.input'][0];ids=c['batch0.ids'][0].tolist();weights=c['batch0.weights'][0].tolist();raw=(root/'batch0.fp16').read_bytes();assert hashlib.sha256(raw).hexdigest()=='c3e5bbedf2cbd7d52ef8c82b21475fd695f3873e0b9703e1e2dea0aa7bc1fc71';recorded=torch.frombuffer(bytearray(raw),dtype=torch.float16)
source=B/'.runs/r262-full-layer-20261028/experts.safetensors';assert source.stat().st_size==453410856
node=next(n for n in ast.parse((R/'research/R156/remote.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='unpack_qpn');env={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<existing-qpn-reference>','exec'),env);unpack=env['unpack_qpn'];lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
src=ast.get_source_segment((R/'research/R156/remote.py').read_text(),node)
src=src.replace('gm=(gs*16384.).to(torch.float16);scale=(scale*gm).to(torch.float16)','gm=gs.to(torch.float64)*16384.;scale=scale.to(torch.float64)*gm')
assert src!=ast.get_source_segment((R/'research/R156/remote.py').read_text(),node)
exec(compile(src,'<no-intermediate-half-rounding>','exec'),env);exact_unpack=env['unpack_qpn']
results=[];qacc=torch.zeros(2048,dtype=torch.float64);oacc=torch.zeros(2048,dtype=torch.float64)
with safe_open(str(source),framework='pt',device='cpu') as f:
 for expert,weight in zip(ids,weights):
  rank,local=divmod(expert,16);art=load_file(str(B/f'.runs/r262-full-layer-20261028/owners/rank{rank}.safetensors'))
  q13=unpack(art['codes13'].reshape(16,-1)[local],art['scales13'].reshape(16,-1)[local],art['gscales13'][local:local+1],1024,2048)
  q2=unpack(art['codes2'].reshape(16,-1)[local],art['scales2'].reshape(16,-1)[local],art['gscales2'][local:local+1],2048,512)
  def original(name):
   p=f'model.language_model.layers.0.mlp.experts.{expert}.{name}.';codes=f.get_tensor(p+'weight');scales=f.get_tensor(p+'weight_scale').double();gs=float(f.get_tensor(p+'weight_scale_2'));n,k=codes.shape;index=torch.stack((codes&15,codes>>4),-1).reshape(n,k*2).long();return (lut[index]*scales.repeat_interleave(16,1)*gs).half().double()
  gate=original('gate_proj');up=original('up_proj');down=original('down_proj');o13=torch.cat((gate,up),0)
  exact13=exact_unpack(art['codes13'].reshape(16,-1)[local],art['scales13'].reshape(16,-1)[local],art['gscales13'][local:local+1],1024,2048);exact2=exact_unpack(art['codes2'].reshape(16,-1)[local],art['scales2'].reshape(16,-1)[local],art['gscales2'][local:local+1],2048,512)
  exact13_diff=int((exact13!=o13).sum());exact2_diff=int((exact2!=down).sum())
  qh=torch.mv(q13,x.double()).half();oh=torch.mv(o13,x.double()).half();qa=(torch.nn.functional.silu(qh[:512])*qh[512:]).half();oa=(torch.nn.functional.silu(oh[:512])*oh[512:]).half();qy=torch.mv(q2,qa.double()).half();oy=torch.mv(down,oa.double()).half();qacc+=qy.double()*weight;oacc+=oy.double()*weight
  results.append({'expert':expert,'weight':weight,'exact_scale_matrix13_changed':exact13_diff,'exact_scale_matrix2_changed':exact2_diff,'matrix13_changed':int((q13!=o13).sum()),'matrix2_changed':int((q2!=down).sum()),'matrix13_max':float((q13-o13).abs().max()),'matrix2_max':float((q2-down).abs().max()),'gateup_max':float((qh.float()-oh.float()).abs().max()),'activation_max':float((qa.float()-oa.float()).abs().max()),'output_max':float((qy.float()-oy.float()).abs().max())})
q=qacc.half();o=oacc.half();d=(q.float()-o.float()).abs();j=int(d.argmax());out={'status':'DIAGNOSTIC','actual_reconstruction_max':float((recorded.float()-q.float()).abs().max()),'original_comparison_max':float(d.max()),'worst_index':j,'actual_value':float(q[j]),'original_value':float(o[j]),'above_threshold':int((d>=0.0001).sum()),'experts':results,'gpu':False};(root/'scale-diagnostic.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
