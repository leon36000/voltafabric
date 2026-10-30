"""Private bounded CPU research multiplexer, one distinct connection per Session."""
import os,pathlib,socket,sys,threading,traceback
import owner_service

def main():
 if len(sys.argv)!=4:raise ValueError('socket, receiver directory and expected sessions required')
 path=pathlib.Path(sys.argv[1]);receiver=pathlib.Path(sys.argv[2]);expected=int(sys.argv[3])
 if expected<1 or expected>40 or not receiver.is_dir():raise ValueError('bad bound')
 srv=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);srv.bind(str(path));os.chmod(path,0o600);srv.listen(expected);srv.settimeout(180)
 threads=[];errors=[]
 def work(conn):
  try:
   with conn,conn.makefile('rb',buffering=0) as inp,conn.makefile('wb',buffering=0) as out:owner_service.serve(inp,out,receiver)
  except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc));traceback.print_exc(file=sys.stderr)
 print('READY',flush=True)
 try:
  for _ in range(expected):
   conn,_=srv.accept();t=threading.Thread(target=work,args=(conn,));t.start();threads.append(t)
 finally:
  srv.close()
  for t in threads:t.join(timeout=180)
 if any(t.is_alive() for t in threads) or errors:raise RuntimeError('terminal failed connection')
 print('CLOSED',flush=True)
if __name__=='__main__':main()
