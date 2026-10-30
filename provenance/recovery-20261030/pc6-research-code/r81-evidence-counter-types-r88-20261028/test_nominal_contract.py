"""Synthetic CPU parser cases, never proof that a worker or GPU ran."""
import copy
import json
from pathlib import Path
import unittest
import nominal_contract as N

ROOT=Path(__file__).resolve().parent


def synthetic():
 raw=(ROOT.parent.parent/'source/fixtures/preweighted_mxfp4_tiny_golden.json').read_bytes();g=json.loads(raw)
 pid=102;runner=101
 fields=['R','101','101','101']+['0']*15+['99']+['0']*5
 stat='102 (CPU_SIMULATED_worker) '+ ' '.join(fields)
 hello=dict(r70_worker_hello=True,pid=pid,runner_pid=runner,rank=7,exe_sha256=N.WORKER,exe=N.WORKER_PATH,
            stat_before=stat,stat_after=stat,hello=dict(pid=pid,rank=7,capabilities=['gpu_mxfp4_r13_preweighted_fp32_v1:'+N.LIBRARY]))
 rows=[hello]
 for batch in [0,1]:
  rows.append(dict(r81_receipt=True,stamp=N.stamp(batch),receipt=dict(rank=7,attempt=batch+1,state='completed',started=True,expected_slots=138,received_slots=138,request_bytes=1,error=None),traffic=dict(loads_transferred=1,execs=batch+1,weight_bytes_sent=6993)))
  rows.append(dict(r81_collector=True,batch=batch,output_bits=N.expected_collector(g)))
 audits=[dict(gpu_audit='startup',library_sha256=N.LIBRARY,lifecycle=[0,0,0],cuda_api_calls=0),
         dict(gpu_audit='evicted',producer={'gpu_r13_resident_fp32_v1':{'library_sha256':N.LIBRARY}},
              lifecycle=[0,0,0],cuda_api_calls=4845,binding=dict(artifact_sha256=N.ARTIFACT_SHA,model_revision='r13-tiny-independent-golden',epoch=20,rank=7,layer=0,experts=[dict(expert=e,local=l)for e,l in [(17,4),(93,1),(241,9)]]),experts=[dict(N.OWNER_COUNTERS,expert=e)for e in [17,93,241]])]
 rows.append(dict(r81_close=True,close=dict(pid=pid,rank=7,exit_code=0,bye_execs=2,killed=False,error=None,stderr='\n'.join(json.dumps(x)for x in audits))))
 rows.append(dict(test='tiny45_twice',execs=2,slots_per_exec=138,upload_plane_copies_per_expert=6,kernel_launches=[736,736,736],cuda_api_calls=4845))
 return rows,raw


class Nominal(unittest.TestCase):
 def test_synthetic_CPU_rows_only(self):
  rows,g=synthetic();v=N.admit_rows(rows,g);self.assertEqual(v['status'],'R81_ROWS_ADMITTED_NOT_COMPLETE_HARDWARE_QUALIFICATION');self.assertTrue(v['guard_exit_and_postflight_still_required'])
  raw=b'test CPU_SIMULATED ... '+b'\n'.join(json.dumps(x).encode()for x in rows)+b'\n'
  self.assertEqual(N.rows_from_log(raw),rows)

 def test_identity_stamp_receipt_order_and_outputs_refused(self):
  rows,g=synthetic()
  mutations=[([0,'exe_sha256'],'0'*64),([0,'hello','capabilities'],[]),([0,'runner_pid'],1),
             ([1,'stamp','run_id'],'old'),([1,'stamp','layer'],False),([1,'receipt','attempt'],2),
             ([1,'receipt','state'],'unknown'),([1,'receipt','started'],False),([1,'receipt','received_slots'],137),
             ([3,'traffic','loads_transferred'],2),([3,'traffic','weight_bytes_sent'],6688),
             ([2,'output_bits',0,0],rows[2]['output_bits'][0][0]^1),([4,'batch'],0),
             ([5,'close','bye_execs'],1),([5,'close','exit_code'],None),([5,'close','killed'],True),
             ([6,'kernel_launches'],[828,828,828]),([6,'cuda_api_calls'],5121)]
  for path,value in mutations:
   with self.subTest(path=path):
    bad=copy.deepcopy(rows);node=bad
    for key in path[:-1]:node=node[key]
    node[path[-1]]=value
    with self.assertRaises(AssertionError):N.admit_rows(bad,g)
  for name in N.OWNER_COUNTERS:
   with self.subTest(counter=name):
    bad=copy.deepcopy(rows);lines=bad[5]['close']['stderr'].splitlines();audit=json.loads(lines[1]);audit['experts'][0][name]+=1;lines[1]=json.dumps(audit);bad[5]['close']['stderr']='\n'.join(lines)
    with self.assertRaises(AssertionError):N.admit_rows(bad,g)

 def test_startup_lifecycle_strict(self):
  rows,g=synthetic()
  for invalid in [[False,False,False],[0.0,0,0]]:
   bad=copy.deepcopy(rows);lines=bad[5]['close']['stderr'].splitlines();audit=json.loads(lines[0]);audit['lifecycle']=invalid;lines[0]=json.dumps(audit);bad[5]['close']['stderr']='\n'.join(lines)
   with self.subTest(invalid=invalid):
    with self.assertRaises(AssertionError):N.admit_rows(bad,g)

 def test_golden_exact_consumed_bytes(self):
  rows,g=synthetic()
  with self.assertRaises(AssertionError):N.admit_rows(rows,g+b' ')
  with self.assertRaises(AssertionError):N.admit_rows(rows,json.loads(g))

 def test_caps_duplicates_and_missing_or_extra_rows(self):
  rows,g=synthetic()
  for bad in [rows[:-1],rows+[rows[0]]]:
   with self.assertRaises(AssertionError):N.admit_rows(bad,g)
  with self.assertRaises(AssertionError):N.strict('{"x":0,"x":1}')
  with self.assertRaises(AssertionError):N.rows_from_log(b'x'*(2097152+1))
  with self.assertRaises(AssertionError):N.rows_from_log(b'{}\n')


if __name__=='__main__':unittest.main()
