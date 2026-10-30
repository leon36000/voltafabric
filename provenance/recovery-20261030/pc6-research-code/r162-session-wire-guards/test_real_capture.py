import io,json,hashlib,pathlib,struct,unittest,wire
p=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/r160-session-wire-cpu');plan=json.loads((p/'plan.json').read_bytes());r=json.loads((p/'result.json').read_bytes());assert r['status']=='WIRE_CAPTURE_VERIFIED' and r['exit']==0
class CaptureTests(unittest.TestCase):
 def test_real_activation_and_header_hash(self):
  body=struct.pack('<2048e',*[((i%31)-15)/16 for i in range(2048)])
  h={'op':'exec_raw_fp16','artifact_sha256':plan['artifact_sha256'],'dispatch':plan['dispatch'],'attempt':1,'inputs':[{'token':0,'values':2048}],'activation_bytes':len(body),'raw_completions':True,'raw_completion_fp16_batch':True}
  b=io.BytesIO();wire.write_frame(b,h,body);encoded=b.getvalue();head,content=encoded.split(b'\n',1);entry=next(x for x in r['steps'] if x.get('op')=='exec_raw_fp16');self.assertEqual(hashlib.sha256(head+b'\n').hexdigest(),entry['header_sha256']);self.assertEqual(hashlib.sha256(content).hexdigest(),entry['body_sha256']);b.seek(0);parsed=wire.read_header(b);tokens,rows=wire.decode_fp16_inputs(parsed,wire.read_exact(b,parsed['activation_bytes']));self.assertEqual(tokens,[0]);self.assertEqual(rows[0],[((i%31)-15)/16 for i in range(2048)])
 def test_short_writes_are_finished(self):
  class Partial(io.BytesIO):
   def write(self,b):return super().write(b[:3])
  s=Partial();wire.write_frame(s,{'reply':'bye','execs':1},b'12345');s.seek(0);self.assertEqual(wire.read_header(s)['execs'],1);self.assertEqual(wire.read_exact(s,5),b'12345')
 def test_nested_duplicate_key_rejected(self):
  with self.assertRaises(wire.WireError):wire.read_header(io.BytesIO(b'{"inputs":[{"token":0,"token":1}]}\n'))
if __name__=='__main__':unittest.main(verbosity=2)
