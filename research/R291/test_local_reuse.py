import hashlib,os,pathlib,tempfile,unittest
from local_reuse import LocalReuse
from adapters.resident_multipart import MultipartError

class LocalReuseTest(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(dir=os.environ['VF_TEST_DIR']);self.root=pathlib.Path(self.t.name);self.source=self.root/'artifact';self.data=b'original bytes';self.source.write_bytes(self.data);self.sha=hashlib.sha256(self.data).hexdigest();self.id={'run_id':'run','model_revision':'model','plan_id':'plan','layer':8,'epoch':1,'rank':2,'attempt':1};self.approved={'path':str(self.source),'bytes':len(self.data),'sha256':self.sha}
 def tearDown(self):self.t.cleanup()
 def create(self):return LocalReuse(self.root,self.id,len(self.data),self.sha,self.approved)
 def test_success_and_claim_is_exclusive(self):
  r=self.create();r.append(0,self.data[:3]);r.append(3,self.data[3:]);self.assertEqual(r.finish(lambda p,i:self.assertEqual((p.read_bytes(),i),(self.data,self.id))),self.source);self.assertTrue((r.directory/'resident.json').exists());self.assertFalse((r.directory/'staged.bin').exists())
  with self.assertRaises(FileExistsError):self.create()
 def test_bad_stream_terminal(self):
  r=self.create();r.append(0,b'x'*len(self.data))
  with self.assertRaises(MultipartError):r.finish(lambda p,i:None)
  with self.assertRaises(MultipartError):r.append(0,self.data)
 def test_wrong_offset_terminal(self):
  r=self.create()
  with self.assertRaises(MultipartError):r.append(1,b'x')
  with self.assertRaises(MultipartError):r.append(0,self.data)
 def test_mutated_original_rejected(self):
  r=self.create();r.append(0,self.data);self.source.write_bytes(b'x'*len(self.data))
  with self.assertRaises(MultipartError):r.finish(lambda p,i:None)
  self.assertFalse((r.directory/'resident.json').exists())
 def test_validator_mutation_rejected(self):
  r=self.create();r.append(0,self.data)
  with self.assertRaises(MultipartError):r.finish(lambda p,i:p.write_bytes(b'x'*len(self.data)))
 def test_symlink_rejected(self):
  self.source.unlink();self.source.symlink_to('/dev/null');r=self.create();r.append(0,self.data)
  with self.assertRaises(OSError):r.finish(lambda p,i:None)
 def test_wrong_approval_rejected(self):
  self.approved['sha256']='0'*64
  with self.assertRaises(MultipartError):self.create()
if __name__=='__main__':unittest.main()
