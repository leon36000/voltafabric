"""Pure R81 nominal-row admission. Synthetic tests are never GPU qualification."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
NAMESPACE='r53-r81-v100-hostinput-session-20261027-a'
LIBRARY='d1a5b39fe3f0ec6519a0e33cdcda80832650cf82db28f678ba516e6a22e90176'
WORKER='3c1fd0dc613b2e0f4caf94e563721715752403767ec75bbca4f1b9e6abaa6823'
WORKER_PATH='/home/ubuntu/NAOS/60_travaux/VoltaFabric-r15-20261024/r53-r61-r80-hostinput-worker-20261027-a/bin/x86_64/worker_release'
TEST='native_worker_matches_all_45_independent_golden_slots_twice_and_sums_routes'
GOLDEN_SHA='51e4c60545c4f29dcbe03d57f017e522095ec2d37e19f67d2fe890670227d1ed'
ARTIFACT_SHA='f0bfbf4220713b7aae0200e15f5232d118f0abfc65d7c322a6540a837fb279a7'
OWNER_COUNTERS=dict(weight_upload_bytes=1632,weight_upload_copies=6,resident_weight_bytes=1632,
 input_h2d_bytes=3036,output_d2h_bytes=11776,status_d2h_bytes=736,exec_calls=92,published=92,
 rejected=0,kernel_launches=736,down_launches=92,device_allocations=16,scratch_device_bytes=966,poisoned=0)


def strict(raw):
 def pairs(items):
  d={}
  for k,v in items:
   assert k not in d,'duplicate JSON key';d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs)


def integer(v):
 assert type(v)is int
 return v


def stamp(batch):
 return dict(run_id=NAMESPACE,model_revision='r13-tiny-independent-golden',plan_id='tiny-45-independent-cases',layer=0,batch=batch,epoch=20)


def expected_collector(g):
 # Reuse the already qualified independent Fraction/RN32 addition, with its
 # pinned loader. No CPU/GPU model is executed here.
 path=ROOT.parent/'gpu-worker-ssh-boundary-r55/verify_wire.py'
 manifest=path.parent/'snapshot/MANIFEST.json'
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='014b00a09ad2d8b2d48fb7d6de0db62babc97de6285656f785cbb22ae4c64cd7'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==strict(manifest.read_bytes())['files']['verify_wire.py']
 spec=importlib.util.spec_from_file_location('r81_frozen_r55_oracle',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 rows=[[int(x,16)for x in case['output_bits']]for case in g['cases']]
 assert len(rows)==45 and all(len(row)==32 for row in rows)
 total=[0]*32
 for expert in [17,93,241]:
  case=next(c for c in g['cases']if c['expert']==expert and c['input']=='pattern' and c['coefficient']==0.375)
  total=[module.add_bits(a,int(b,16))for a,b in zip(total,case['output_bits'])]
 return rows+[total]


def rows_from_log(raw):
 assert type(raw)is bytes and len(raw)<=2097152
 rows=[]
 markers=[b'"r70_worker_hello"',b'"r81_receipt"',b'"r81_collector"',b'"r81_close"',b'"tiny45_twice"']
 for line in raw.splitlines():
  if any(mark in line for mark in markers):
   assert b'{'in line and len(line)<=131072
   rows.append(strict(line[line.index(b'{'):]))
 assert len(rows)==7
 return rows


def admit_rows(rows,g):
 assert type(g)is bytes and len(g)<=1048576 and hashlib.sha256(g).hexdigest()==GOLDEN_SHA
 g=strict(g);assert g["weights_sha256"]==ARTIFACT_SHA and g["case_count"]==45
 assert len(rows)==7
 def select(key):return [r for r in rows if r.get(key)is True]
 hello=select('r70_worker_hello');receipts=select('r81_receipt');collectors=select('r81_collector');closed=select('r81_close')
 assert [len(hello),len(receipts),len(collectors),len(closed)]==[1,2,2,1]
 summary=[r for r in rows if r.get('test')=='tiny45_twice'];assert len(summary)==1
 h=hello[0];pid=integer(h['pid']);runner=integer(h['runner_pid']);assert pid>0 and runner>0 and pid!=runner
 assert integer(h['hello']['rank'])==7 and 'gpu_mxfp4_r13_preweighted_fp32_v1:'+LIBRARY in h['hello']['capabilities']
 assert integer(h['hello']['pid'])==pid and integer(h['rank'])==7 and h['exe_sha256']==WORKER and h['exe']==WORKER_PATH
 before=h['stat_before'].rsplit(') ',1)[1].split();after=h['stat_after'].rsplit(') ',1)[1].split()
 assert before[19]==after[19] and before[1]==after[1]==str(runner)
 assert before[2]==after[2]==str(runner) and before[3]==after[3]==str(runner)
 expected=expected_collector(g)
 assert [integer(x['stamp']['batch'])for x in receipts]==[0,1]
 assert [integer(x['batch'])for x in collectors]==[0,1]
 for batch,row in enumerate(receipts):
  assert row['stamp']==stamp(batch) and all(type(row['stamp'][k])is int for k in ['layer','batch','epoch'])
  v=row['receipt'];assert integer(v['rank'])==7 and integer(v['attempt'])==batch+1
  assert v['state']=='completed' and v['started']is True and v['error']is None
  assert integer(v['expected_slots'])==integer(v['received_slots'])==138
  traffic=row['traffic'];assert integer(traffic['loads_transferred'])==1 and integer(traffic['execs'])==batch+1
  assert integer(traffic['weight_bytes_sent'])==6993
  output=collectors[batch]['output_bits'];assert output==expected and all(type(bit)is int for r in output for bit in r)
 close=closed[0]['close'];assert integer(close['pid'])==pid and integer(close['rank'])==7
 assert close['exit_code']==0 and type(close['exit_code'])is int and integer(close['bye_execs'])==2 and close['killed']is False and close['error']is None
 audits=[]
 for line in close['stderr'].splitlines():
  if line.startswith('{'):audits.append(strict(line))
 startup=[a for a in audits if a.get('gpu_audit')=='startup'];evicted=[a for a in audits if a.get('gpu_audit')=='evicted']
 assert len(startup)==len(evicted)==1
 assert startup[0]['library_sha256']==LIBRARY and startup[0]['lifecycle']==[0,0,0] and all(type(v)is int for v in startup[0]['lifecycle']) and integer(startup[0]['cuda_api_calls'])==0
 e=evicted[0];assert e['producer']=={'gpu_r13_resident_fp32_v1':{'library_sha256':LIBRARY}}
 assert e['lifecycle']==[0,0,0] and all(type(v)is int for v in e['lifecycle']) and integer(e['cuda_api_calls'])==4845
 binding=e['binding'];assert binding['artifact_sha256']==ARTIFACT_SHA and binding['model_revision']=='r13-tiny-independent-golden'
 assert integer(binding['epoch'])==20 and integer(binding['rank'])==7 and integer(binding['layer'])==0
 assert binding['experts']==[dict(expert=x,local=y)for x,y in [(17,4),(93,1),(241,9)]]
 assert all(type(item['expert']) is int and type(item['local']) is int for item in binding['experts'])
 owners=e['experts'];assert len(owners)==3 and sorted(integer(o['expert'])for o in owners)==[17,93,241]
 for owner in owners:
  for name,value in OWNER_COUNTERS.items():assert integer(owner[name])==value,(name,owner[name],value)
 s=summary[0];assert integer(s['execs'])==2 and integer(s['slots_per_exec'])==138 and integer(s['upload_plane_copies_per_expert'])==6
 assert [integer(value) for value in s['kernel_launches']]==[736,736,736] and integer(s['cuda_api_calls'])==4845
 return dict(status='R81_ROWS_ADMITTED_NOT_COMPLETE_HARDWARE_QUALIFICATION',native_pid=pid,runner_pid=runner,
             exact_Collector_values=2944,input_h2d_bytes=9108,slots=276,kernels=2208,cuda_api_calls=4845,
             guard_exit_and_postflight_still_required=True)
