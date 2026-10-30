"""Fresh local or SSH stdio endpoint for a CPU-only LocalReuse owner."""
import os,pathlib,sys
if len(sys.argv)!=2:raise ValueError('receiver required')
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from adapters.resource_admission import ResourcePolicy
ResourcePolicy(2*1024**3,1024**3,2*1024**3,256*1024**2).check(pathlib.Path(sys.argv[1]))
import owner_service
owner_service.serve(sys.stdin.buffer,sys.stdout.buffer,pathlib.Path(sys.argv[1]))
