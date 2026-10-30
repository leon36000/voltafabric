import ast, copy, hashlib, json, pathlib, subprocess, sys, unittest
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from admit import admit, AdmissionError
BASE=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012')
class DonorIntegration(unittest.TestCase):
 def test_actual_donor_proposals(self):
  source=BASE/'.runs/deepinfer-source-audit-r163-20261028/source/deepinfer/model_executor/eplb/policy.py';raw=source.read_bytes();receipt=json.loads((BASE/'.runs/deepinfer-eplb-routing-r172-20261028/receipt.json').read_text());self.assertEqual(hashlib.sha256(raw).hexdigest(),receipt['upstream_sha256']);nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('_batch_greedy_indices','rebalance_experts_global')];env={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env);torch.set_num_threads(1);self.assertIsNone(torch.version.cuda)
  problem={'hidden':1,'experts':[0,1,2,3],'routes':[[{'expert':0,'weight':0.25},{'expert':1,'weight':0.75}]],'registry':{'epoch':8,'copies':[]},'stamp':{'run_id':'r177-donor','model_revision':'synthetic','plan_id':'candidate','layer':0,'batch':0,'epoch':8},'preferred_owners':{},'resident_order':{}}
  for name,counts in [('balanced',[10,10,10,10]),('hot',[100,20,10,5])]:
   physical,_,_=env['rebalance_experts_global'](torch.tensor([[counts]],dtype=torch.float32),6,3);seq=physical.tolist()[0];proposal={'rank_ids':[2,9],'capacity_bytes':{2:30,9:30},'resident_bytes':{0:10,1:10,2:10,3:10},'proposed':[seq[:3],seq[3:]]}
   if name=='hot':
    with self.assertRaises(AdmissionError):admit(problem,proposal)
   else:
    value=admit(problem,proposal);self.assertEqual(value['routes'],problem['routes']);cli=BASE/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';self.assertEqual(hashlib.sha256(cli.read_bytes()).hexdigest(),'11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4');r=subprocess.run([str(cli),'routing','-'],input=json.dumps(value).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10);self.assertEqual(r.returncode,0,r.stdout);self.assertTrue(json.loads(r.stdout)['ok'])
if __name__=='__main__':unittest.main()
