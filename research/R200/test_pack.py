import hashlib,json,pathlib,sys,tempfile,unittest
P=pathlib.Path(__file__).resolve().parent;R=P.parent.parent;B=R.parent.parent;sys.path.insert(0,str(P));from pack_checkpoint import pack
class PackTest(unittest.TestCase):
 def test_real_expert_export_and_hash_refusal(self):
  source=R/'fixtures/checkpoint_qwen36_expert0_full.safetensors';sha=hashlib.sha256(source.read_bytes()).hexdigest();cli=B/'gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';cli_sha='11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4'
  with tempfile.TemporaryDirectory(dir=P) as d:
   args=dict(source=source,source_sha=sha,output=pathlib.Path(d)/'out',revision='test-real-fixture',layer=0,epoch=1,groups={2:[0]},cli=cli,cli_sha=cli_sha);result=pack(**args);self.assertEqual(result['2']['experts'],[0]);self.assertGreater(result['2']['size'],1700000);self.assertEqual(json.loads((args['output']/'manifest.json').read_bytes())['ranks']['2']['sha256'],result['2']['sha256'])
   with self.assertRaises(FileExistsError):pack(**args)
   args.update(source_sha='0'*64,output=pathlib.Path(d)/'bad')
   with self.assertRaises(ValueError):pack(**args)
   self.assertFalse(args['output'].exists())
 def test_duplicate_expert_refused(self):
  source=R/'fixtures/checkpoint_qwen36_expert0_full.safetensors';sha=hashlib.sha256(source.read_bytes()).hexdigest();cli=B/'gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli'
  with tempfile.TemporaryDirectory(dir=P) as d:
   with self.assertRaises(ValueError):pack(source=source,source_sha=sha,output=pathlib.Path(d)/'bad',revision='x',layer=0,epoch=1,groups={2:[0],9:[0]},cli=cli,cli_sha='11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4')
if __name__=='__main__':unittest.main()
