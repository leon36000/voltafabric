import pathlib,sys,unittest
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));from admission import filter_from_bindings,AdmissionError
class TestAdmission(unittest.TestCase):
 def setUp(self):
  self.args=dict(model_revision='model-sha',plan_id='plan-sha',epoch=3,layers=[0,2],expert_count=4,remote_ids={1,3},load_format='safetensors',strategy='lazy',multithread=False,prefetch=False,eplb=False,bindings=[dict(model_revision='model-sha',plan_id='plan-sha',epoch=3,layer=l,expert=e,rank=2,artifact_sha256='a'*64,state='READY') for l in [0,2] for e in [1,3]])
 def test_intersection_keeps_local_and_empty_remote(self):
  self.assertEqual(filter_from_bindings(**self.args),{0,2});self.args['remote_ids']=set();self.assertEqual(filter_from_bindings(**self.args),{0,1,2,3})
 def test_reject_missing_or_foreign(self):
  self.args['bindings'].pop()
  with self.assertRaises(AdmissionError):filter_from_bindings(**self.args)
  self.args['bindings'][0]['epoch']=4
  with self.assertRaises(AdmissionError):filter_from_bindings(**self.args)
 def test_reject_unsupported_loader_and_eplb(self):
  for key,value in [('strategy','eager'),('prefetch',True),('multithread',True),('load_format','fastsafetensors'),('eplb',True)]:
   args=dict(self.args);args[key]=value
   with self.assertRaises(AdmissionError):filter_from_bindings(**args)
 def test_reject_duplicate_or_unready_binding(self):
  self.args['bindings'].append(self.args['bindings'][0].copy())
  with self.assertRaises(AdmissionError):filter_from_bindings(**self.args)
  self.args['bindings'].pop();self.args['bindings'][0]['state']='UNKNOWN'
  with self.assertRaises(AdmissionError):filter_from_bindings(**self.args)
if __name__=='__main__':unittest.main()
