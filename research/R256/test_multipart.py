import hashlib,json,pathlib,struct,sys,unittest,uuid
ROOT=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012')
sys.path.insert(0,str(ROOT/'.runs/gitlab-import-r113-20261028/repository'))
from adapters.resident_multipart import ResidentMultipart,MultipartError,MAX_CHUNK_BYTES
from scripts.resident_artifact import load_stack

class Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=ROOT/'.runs'/('r256-multipart-receiver-'+uuid.uuid4().hex);cls.root.mkdir(mode=0o700)
 def identity(self):return {'run_id':'r256-'+uuid.uuid4().hex,'model_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','plan_id':'r256-cpu-receiver','layer':0,'epoch':1,'rank':2,'attempt':1}
 def make(self,data,identity=None,digest=None):return ResidentMultipart(self.root,identity or self.identity(),len(data),digest or hashlib.sha256(data).hexdigest())
 def test_real_ten_expert_artifact_chunked_and_validated(self):
  path=ROOT/'.runs/r255-artifact-frame-20261028/ten/rank2.safetensors';data=path.read_bytes();self.assertEqual(hashlib.sha256(data).hexdigest(),'b97f59a4bf9e9720a97d3b546e53f1134d613c49a70bcc47ef15852b82ce0e46');identity=self.identity();r=self.make(data,identity)
  for offset in range(0,len(data),MAX_CHUNK_BYTES):r.append(offset,data[offset:offset+MAX_CHUNK_BYTES])
  def validate(p,bind):
   with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];meta=json.loads(f.read(n))['__metadata__'];info=json.loads(meta['vf_info'])
   self.assertEqual(meta['vf_schema'],'QPN2_RESIDENT_V1');self.assertEqual((info['model_revision'],info['layer'],info['rank'],info['epoch']),(bind['model_revision'],bind['layer'],bind['rank'],bind['epoch']));self.assertEqual(len(info['physical_experts']),10)
   loaded=load_stack(p,expected=info,expected_sha256='b97f59a4bf9e9720a97d3b546e53f1134d613c49a70bcc47ef15852b82ce0e46');self.assertEqual(loaded.metadata,info)
  out=r.finish(validate);self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(),hashlib.sha256(data).hexdigest())
  with self.assertRaises(FileExistsError):self.make(data,identity)
  with self.assertRaises(MultipartError):r.append(len(data),b'x')
 def test_order_duplicate_and_terminal(self):
  r=self.make(b'abcd');r.append(0,b'ab')
  with self.assertRaises(MultipartError):r.append(0,b'ab')
  with self.assertRaises(MultipartError):r.finish(lambda p,i:None)
  self.assertFalse((r.directory/'resident.bin').exists())
 def test_incomplete_and_digest(self):
  r=self.make(b'abcd');r.append(0,b'ab')
  with self.assertRaises(MultipartError):r.finish(lambda p,i:None)
  s=self.make(b'abcd',digest='0'*64);s.append(0,b'abcd')
  with self.assertRaises(MultipartError):s.finish(lambda p,i:None)
  self.assertFalse((s.directory/'resident.bin').exists())
 def test_validator_reject_and_mutation(self):
  r=self.make(b'abc');r.append(0,b'abc')
  def reject(p,i):raise RuntimeError('invalid resident')
  with self.assertRaises(RuntimeError):r.finish(reject)
  self.assertFalse((r.directory/'resident.bin').exists())
  s=self.make(b'abc');s.append(0,b'abc')
  def mutate(p,i):p.write_bytes(b'def')
  with self.assertRaises(MultipartError):s.finish(mutate)
  self.assertFalse((s.directory/'resident.bin').exists())
 def test_oversized_chunk_consumes_identity(self):
  payload=b'x'*(MAX_CHUNK_BYTES+1); identity=self.identity();r=self.make(payload,identity)
  with self.assertRaises(MultipartError):r.append(0,payload)
  with self.assertRaises(FileExistsError):self.make(payload,identity)
  self.assertFalse((r.directory/'resident.bin').exists())
 def test_bad_bounds(self):
  i=self.identity()
  with self.assertRaises(MultipartError):ResidentMultipart(self.root,i,0,'0'*64)
  with self.assertRaises(MultipartError):ResidentMultipart(self.root,i,256*1024*1024+1,'0'*64)
  i['epoch']=True
  with self.assertRaises(MultipartError):ResidentMultipart(self.root,i,3,'0'*64)
if __name__=='__main__':unittest.main()
