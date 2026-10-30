"""Byte-for-byte Unix socket bridge for one session; no retries."""
import os,socket,sys,threading
if len(sys.argv)!=2:raise ValueError('socket path required')
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(sys.argv[1])
def incoming():
 try:
  while True:
   data=os.read(0,65536)
   if not data:break
   s.sendall(data)
 finally:
  try:s.shutdown(socket.SHUT_WR)
  except OSError:pass
t=threading.Thread(target=incoming,daemon=True);t.start()
while True:
 data=s.recv(65536)
 if not data:break
 view=memoryview(data)
 while view:
  n=os.write(1,view)
  if n<=0:raise OSError('short write')
  view=view[n:]
s.close()
