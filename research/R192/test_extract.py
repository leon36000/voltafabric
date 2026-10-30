import hashlib,json,pathlib,struct,subprocess,sys,tempfile,unittest
P=pathlib.Path(__file__).resolve().parent;R=P.parent.parent;sys.path.insert(0,str(P));from extract_verified import extract
class TestExtract(unittest.TestCase):
 def test_real_fixture_exact_selected_tensors(self):
  src=R/'fixtures/checkpoint_qwen36_expert0_full.safetensors';sha=hashlib.sha256(src.read_bytes()).hexdigest()
  with tempfile.TemporaryDirectory(dir=P) as d:
   dst=pathlib.Path(d)/'selected.safetensors';result=extract(src,sha,dst,[0]);self.assertEqual(result['tensor_count'],9)
   def read(p):
    with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];m=json.loads(f.read(n));base=8+n;return {k:(v['dtype'],v['shape'],(f.seek(base+v['data_offsets'][0]),f.read(v['data_offsets'][1]-v['data_offsets'][0]))[1]) for k,v in m.items() if k!='__metadata__'}
   before,after=read(src),read(dst);self.assertEqual(after,{k:before[k] for k in after});self.assertEqual(result['selected_sha256'],hashlib.sha256(dst.read_bytes()).hexdigest())
   request=json.loads((R/'fixtures/resident_import_qwen_expert0.json').read_bytes());cli=R.parent.parent/'gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';outputs=[]
   for i,source in enumerate((src,dst)):
    request['path']=str(source);out=pathlib.Path(d)/f'resident-{i}.safetensors';body=json.dumps({'source':{'kind':'checkpoint','request':request},'output':str(out)},separators=(',',':')).encode();proc=subprocess.run([str(cli),'resident-export','-'],input=body,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30);self.assertEqual(proc.returncode,0,(proc.stdout,proc.stderr));outputs.append(hashlib.sha256(out.read_bytes()).hexdigest())
   self.assertEqual(outputs[0],outputs[1])
   with self.assertRaises(FileExistsError):extract(src,sha,dst,[0])
 def test_bad_hash_and_missing_tensor_refuse(self):
  src=R/'fixtures/checkpoint_qwen36_expert0_full.safetensors';sha=hashlib.sha256(src.read_bytes()).hexdigest()
  with tempfile.TemporaryDirectory(dir=P) as d:
   dst=pathlib.Path(d)/'x';
   with self.assertRaisesRegex(ValueError,'SHA256'):extract(src,'0'*64,dst,[0])
   self.assertFalse(dst.exists())
   with self.assertRaises(KeyError):extract(src,sha,dst,[1])
   self.assertFalse(dst.exists())
if __name__=='__main__':unittest.main()
