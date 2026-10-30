"""Private PC1 CPU multiplexer; inherited stdin is the control pipe."""
import os,pathlib,socket,sys
from adapters.cpu_multiplex import serve_fleet
import owner_service

def main():
    if len(sys.argv)!=4:
        raise ValueError('socket, receiver directory and session bound required')
    path=pathlib.Path(sys.argv[1]);receiver=pathlib.Path(sys.argv[2]);limit=int(sys.argv[3])
    if not receiver.is_dir() or not 1<=limit<=40:
        raise ValueError('invalid bounds')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
        listener.bind(str(path));os.chmod(path,0o600);listener.listen(limit)
        print('READY',flush=True)
        def serve(conn):
            with conn.makefile('rb',buffering=0) as inp,conn.makefile('wb',buffering=0) as out:
                owner_service.serve(inp,out,receiver)
        count=serve_fleet(listener,sys.stdin.fileno(),serve,limit,180)
        print('CLOSED '+str(count),flush=True)
if __name__=='__main__':main()
