import pathlib,hashlib,json,subprocess,signal,struct,math,os
p=pathlib.Path(__file__).parent;q=(p/'plan.json').read_bytes();assert hashlib.sha256(q).hexdigest()=='914f4ea3dcbd907bb0329d066d0b4c0fe100b5704916a06afcee1bdaf28f56fb';plan=json.loads(q);worker=pathlib.Path(plan['worker']);artifact=pathlib.Path(plan['artifact']).read_bytes();assert hashlib.sha256(worker.read_bytes()).hexdigest()==plan['worker_sha256'] and hashlib.sha256(artifact).hexdigest()==plan['artifact_sha256'];steps=[];proc=None;status='UNKNOWN_NO_REPLAY'
def alarm(*a):raise TimeoutError('wire deadline')
signal.signal(signal.SIGALRM,alarm);signal.alarm(25)
def send(header,body=b''):
 raw=json.dumps(header,separators=(',',':')).encode()+b'\n';assert len(raw)+len(body)<16*1024*1024;proc.stdin.write(raw+body);proc.stdin.flush();steps.append({'direction':'send','op':header['op'],'header_sha256':hashlib.sha256(raw).hexdigest(),'body_bytes':len(body),'body_sha256':hashlib.sha256(body).hexdigest()})
def read():
 line=proc.stdout.readline(1048577);assert line.endswith(b'\n') and len(line)<=1048576;v=json.loads(line);steps.append({'direction':'recv','header':v,'header_sha256':hashlib.sha256(line).hexdigest()});return v
try:
 proc=subprocess.Popen([str(worker)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);(p/'intent.json').write_text(json.dumps({'pid':proc.pid,'plan_sha256':hashlib.sha256(q).hexdigest(),'no_retry':True})+'\n')
 send({'op':'hello','protocol':1,'rank':2});hello=read();assert hello['reply']=='hello' and hello['hello']['rank']==2;caps=hello['hello']['capabilities'];assert 'cpu_qpn2_resident_raw_v1' in caps and 'cpu_exec_fp16_raw_v1' in caps and 'cpu_completion_batch_fp16_raw_v1' in caps
 send({'op':'load_artifact_raw','artifact_sha256':plan['artifact_sha256'],'epoch':1,'artifact_bytes':len(artifact)},artifact);loaded=read();assert loaded['reply']=='loaded' and loaded['transferred'] and loaded['artifact_sha256']==plan['artifact_sha256']
 x=struct.pack('<2048e',*[((i%31)-15)/16 for i in range(2048)]);dispatch=plan['dispatch'];send({'op':'exec_raw_fp16','artifact_sha256':plan['artifact_sha256'],'dispatch':dispatch,'attempt':1,'inputs':[{'token':0,'values':2048}],'activation_bytes':len(x),'raw_completions':True,'raw_completion_fp16_batch':True},x)
 started=read();assert started['reply']=='event' and started['event']['event']['kind']=='started';header=read();assert header['reply']=='completion_batch_fp16_raw' and header['values']==2048 and header['rank']==2 and header['attempt']==1 and header['stamp']==dispatch['stamp'] and header['completion_stamp']==dispatch['stamp'];body=proc.stdout.read(4096);assert len(body)==4096 and all(math.isfinite(x) for x in struct.unpack('<2048e',body));(p/'completion.fp16').write_bytes(body);steps.append({'direction':'recv_body','bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()});done=read();assert done['reply']=='event' and done['event']['event']['kind']=='finished';send({'op':'shutdown'});bye=read();assert bye['reply']=='bye' and bye['execs']==1;proc.stdin.close();exitcode=proc.wait(timeout=3);assert exitcode==0;status='WIRE_CAPTURE_VERIFIED';result={'status':status,'exit':exitcode,'completion_sha256':hashlib.sha256(body).hexdigest(),'completion_values':2048,'steps':steps,'gpu':False,'limits':'Real Rust CPU worker wire only; no Python owner bridge, CUDA or full model.'}
except BaseException as e:
 result={'status':status,'error':type(e).__name__,'reason':str(e)[:500],'steps':steps}
 if proc is not None and proc.poll() is None:proc.kill();proc.wait(timeout=3)
finally:
 signal.alarm(0)
 if proc is not None:result['worker_pid']=proc.pid;result['worker_exit']=proc.returncode;err=proc.stderr.read(65537);(p/'worker.stderr').write_bytes(err);result['stderr_sha256']=hashlib.sha256(err).hexdigest()
 (p/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='steps'}))
