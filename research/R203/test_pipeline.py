import hashlib,json,pathlib,sys,tempfile,unittest
from safetensors import safe_open
from safetensors.torch import save_file
P=pathlib.Path(__file__).resolve().parent;R=P.parent.parent;B=R.parent.parent;sys.path.insert(0,str(P));from run_session import run
class PipelineTest(unittest.TestCase):
 def test_synthetic_eight_copies_end_to_end(self):
  with tempfile.TemporaryDirectory(dir=P) as t:
   root=pathlib.Path(t);src=R/'fixtures/checkpoint_qwen36_expert0_full.safetensors';out=root/'cloned.safetensors';values={}
   with safe_open(str(src),framework='pt') as f:
    for name in f.keys():
     if '.experts.0.' in name:
      value=f.get_tensor(name)
      for e in range(8):values[name.replace('.experts.0.',f'.experts.{e}.')]=value.clone()
    save_file(values,str(out),metadata={'format':'pt'})
   plan={'operation_id':'r203-test-synthetic','source':str(out),'source_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'revision':'r203-synthetic-eight','experts':list(range(8)),'layer':0,'oracle_atol':.0001,'cli_sha256':'11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4','gpu':False};(root/'plan.json').write_text(json.dumps(plan));result=run(root/'plan.json',root/'run');self.assertEqual(result['status'],'SUCCEEDED');self.assertEqual(result['oracle']['expert_ids'],list(range(8)));print(json.dumps({'run_id':result['run_id'],'oracle':result['oracle']['records']}))
if __name__=='__main__':unittest.main()
