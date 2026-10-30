import copy, json, pathlib, subprocess, sys, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from admit import admit, AdmissionError

class TestAdmission(unittest.TestCase):
 def setUp(self):
  self.problem={'hidden':1,'experts':[0,1,2,3],'routes':[[{'expert':0,'weight':0.25},{'expert':1,'weight':0.75}]],'registry':{'epoch':8,'copies':[]},'stamp':{'run_id':'r177-cpu','model_revision':'test','plan_id':'candidate','layer':0,'batch':0,'epoch':8},'preferred_owners':{},'resident_order':{}}
  self.p={'rank_ids':[2,9],'capacity_bytes':{2:30,9:30},'resident_bytes':{0:10,1:10,2:10,3:10},'proposed':[[0,2,3],[1,3,2]]}
 def run_admit(self,p=None,problem=None):return admit(copy.deepcopy(problem or self.problem),copy.deepcopy(p or self.p))
 def test_real_rust_routing_and_preservation(self):
  original=copy.deepcopy(self.problem);out=self.run_admit();self.assertEqual(out['routes'],original['routes']);self.assertEqual(out['stamp'],original['stamp']);self.assertEqual(out['registry']['epoch'],8);self.assertEqual(self.problem,original)
  cli='/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';r=subprocess.run([cli,'routing','-'],input=json.dumps(out).encode(),capture_output=True,timeout=10);self.assertEqual(r.returncode,0,r.stdout);data=json.loads(r.stdout);slots=[x['slot'] for rank in data['result']['ranks'] for x in rank['slots']];self.assertEqual(sorted((x['expert'],x['weight']) for x in slots),[(0,0.25),(1,0.75)])
 def test_hot_duplicate_rejected_atomically(self):
  self.p['proposed']=[[0,0,3],[0,1,2]];original=copy.deepcopy(self.problem)
  with self.assertRaises(AdmissionError):self.run_admit()
  self.assertEqual(self.problem,original)
 def test_capacity_and_coverage(self):
  self.p['capacity_bytes'][2]=29
  with self.assertRaises(AdmissionError):self.run_admit()
  self.p['capacity_bytes'][2]=30;self.p['proposed']=[[0,2,3],[1,3,3]]
  with self.assertRaises(AdmissionError):self.run_admit()
  self.p['proposed']=[[0,2,3],[1,3,8]]
  with self.assertRaises(AdmissionError):self.run_admit()
 def test_identity_and_type_refusals(self):
  x=copy.deepcopy(self.problem);x['stamp']['epoch']=9
  with self.assertRaises(AdmissionError):self.run_admit(problem=x)
  x=copy.deepcopy(self.problem);x['stamp']['epoch']=True;x['registry']['epoch']=1
  with self.assertRaises(AdmissionError):self.run_admit(problem=x)
  self.p['rank_ids']=[2,2]
  with self.assertRaises(AdmissionError):self.run_admit()
  self.p['rank_ids']=[2,9];self.p['resident_bytes'][0]=True
  with self.assertRaises(AdmissionError):self.run_admit()
 def test_preferred_owner_must_be_covered(self):
  x=copy.deepcopy(self.problem);x['preferred_owners']={"0":9}
  with self.assertRaises(AdmissionError):self.run_admit(problem=x)
 def test_repeat_deterministic(self):self.assertEqual(self.run_admit(),self.run_admit())
if __name__=='__main__':unittest.main()
