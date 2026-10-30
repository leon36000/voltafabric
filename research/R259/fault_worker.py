import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);mode=(root/'mode.txt').read_text().strip()
assert json.loads(sys.stdin.readline())=={'op':'hello','protocol':1,'rank':2}
caps=['cpu_f64_dense_moe_v1']
if mode=='bad_ack':caps.append('cpu_qpn2_multipart_v1')
print(json.dumps({'reply':'hello','hello':{'protocol':1,'rank':2,'arch':'aarch64','os':'linux','endian':'little','build':'fault-fixture','capabilities':caps,'max_message_bytes':16777216,'pid':__import__('os').getpid()}}),flush=True)
request=json.loads(sys.stdin.readline())
if mode=='missing_capability':
 assert request=={'op':'shutdown'}
 (root/'evidence.json').write_text(json.dumps({'only_shutdown_after_hello':True}))
 print(json.dumps({'reply':'bye','execs':0}),flush=True)
else:
 assert request['op']=='begin_multipart'
 print(json.dumps({'reply':'multipart_ready','offset':1}),flush=True)
