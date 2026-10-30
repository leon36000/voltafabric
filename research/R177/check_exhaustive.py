"""Independent finite enumeration for the bounded R172 placement domain."""
import itertools,json,pathlib,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent));from admit import admit,AdmissionError
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');cli=B/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli'
problem={'hidden':1,'experts':[0,1,2,3],'routes':[[{'expert':0,'weight':0.25},{'expert':1,'weight':0.75}]],'registry':{'epoch':8,'copies':[]},'stamp':{'run_id':'r177-exhaustive','model_revision':'synthetic','plan_id':'enum','layer':0,'batch':0,'epoch':8},'preferred_owners':{},'resident_order':{}}
ok=bad=rust=0
for flat in itertools.product(range(4),repeat=6):
 expected=len(set(flat[:3]))==3 and len(set(flat[3:]))==3 and set(flat)==set(range(4))
 proposal={'rank_ids':[2,9],'capacity_bytes':{2:30,9:30},'resident_bytes':{i:10 for i in range(4)},'proposed':[list(flat[:3]),list(flat[3:])]}
 try: result=admit(problem,proposal);accepted=True
 except AdmissionError:accepted=False
 assert accepted==expected,(flat,accepted,expected)
 if accepted:
  ok+=1
  if rust<16:
   p=subprocess.run([str(cli),'routing','-'],input=json.dumps(result).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10);assert p.returncode==0 and json.loads(p.stdout)['ok'];rust+=1
 else:bad+=1
print(json.dumps({'total':ok+bad,'accepted':ok,'rejected':bad,'actual_rust_samples':rust,'gpu':False}))
