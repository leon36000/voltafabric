import json,os,sys
h=json.loads(sys.stdin.readline());rank=h['rank'];assert h['op']=='hello'
print(json.dumps({'reply':'hello','hello':{'protocol':1,'rank':rank,'arch':'aarch64','os':'linux','endian':'little','build':'legacy-test','capabilities':['cpu_f64_dense_moe_v1','cpu_qpn2_resident_raw_v1'],'max_message_bytes':16777216,'pid':os.getpid()}}),flush=True)
sys.stdin.buffer.read()
