import io,struct,unittest
import wire
class SessionWireTests(unittest.TestCase):
 def test_valid_fragmented_body_and_raw_fp16(self):
  class Fragment(io.BytesIO):
   def read(self,n=-1):return super().read(min(n,3) if n>=0 else 3)
  data=b'{"op":"exec_raw_fp16","activation_bytes":8,"inputs":[{"token":2,"values":2},{"token":7,"values":2}]}\n'+struct.pack('<4e',1.,-2.,0.5,4.)
  f=Fragment(data);h=wire.read_header(f);raw=wire.read_exact(f,h['activation_bytes']);self.assertEqual(wire.decode_fp16_inputs(h,raw),([2,7],[[1.,-2.],[.5,4.]]))
 def test_duplicate_json_and_truncated_header(self):
  for data in [b'{"op":"a","op":"b"}\n',b'{"op":"a"}']:
   with self.subTest(data=data),self.assertRaises(wire.WireError):wire.read_header(io.BytesIO(data))
 def test_body_short_and_oversize(self):
  with self.assertRaises(wire.WireError):wire.read_exact(io.BytesIO(b'a'),2)
  with self.assertRaises(wire.WireError):wire.read_exact(io.BytesIO(),wire.MAX_BODY+1)
 def test_nonfinite_bad_sizes_and_duplicate_tokens(self):
  cases=[({'activation_bytes':2,'inputs':[{'token':0,'values':1}]},struct.pack('<H',0x7c00)),({'activation_bytes':4,'inputs':[{'token':0,'values':1}]},b'\0\0\0\0'),({'activation_bytes':4,'inputs':[{'token':0,'values':1},{'token':0,'values':1}]},b'\0\0\0\0'),({'activation_bytes':2,'inputs':[{'token':False,'values':1}]},b'\0\0')]
  for h,b in cases:
   with self.subTest(h=h),self.assertRaises(wire.WireError):wire.decode_fp16_inputs(h,b)
 def test_frame_roundtrip_and_bounded_header(self):
  b=io.BytesIO();wire.write_frame(b,{'reply':'bye','execs':2},b'\x01\x02');b.seek(0);self.assertEqual(wire.read_header(b),{'reply':'bye','execs':2});self.assertEqual(wire.read_exact(b,2),b'\x01\x02')
  with self.assertRaises(wire.WireError):wire.write_frame(io.BytesIO(),{'x':'a'*wire.MAX_HEADER})
if __name__=='__main__':unittest.main(verbosity=2)
