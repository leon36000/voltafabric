"""Research-only shared-expert order adapter for the pinned 1Cat call boundary."""
import pathlib,sys,threading
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent/'R182'))
from durable_hook import DurableSession,DurableHookError

class SharedAwareSession(DurableSession):
 def __init__(self,*args,internal_order,**kwargs):
  super().__init__(*args,**kwargs);self.internal_order=internal_order;self.lock=threading.RLock()
 def __call__(self,layer,x,weights,ids,shared_experts,shared_experts_input):
  with self.lock:
   if self.terminal:raise DurableHookError('terminal session; no replay')
   if shared_experts is None:
    if shared_experts_input is not None:raise DurableHookError('shared input without shared expert')
    return super().__call__(layer,x,weights,ids,None,None)
   # Reproduce the parent's preflight before the shared module is allowed to run.
   if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float16 or tuple(x.shape)!=(1,2048) or not torch.isfinite(x).all()
       or not isinstance(ids,torch.Tensor) or ids.device.type!='cpu' or ids.dtype not in (torch.int32,torch.int64) or ids.tolist()!=[[0,1]]
       or not isinstance(weights,torch.Tensor) or weights.device.type!='cpu' or not weights.is_floating_point() or tuple(weights.shape)!=(1,2) or not torch.isfinite(weights).all() or (weights<0).any()
       or not isinstance(shared_experts_input,torch.Tensor) or shared_experts_input.device.type!='cpu' or not torch.isfinite(shared_experts_input).all()):
    raise DurableHookError('invalid shared or routed input')
   try:
    # Actual 1Cat SharedExperts.apply selects the appropriate order and owns its
    # output.  The runner, not this hook, subsequently consumes/combines it.
    shared_experts.apply(shared_experts_input,self.internal_order)
    return super().__call__(layer,x,weights,ids,None,None)
   except BaseException:
    self.terminal=True
    raise
