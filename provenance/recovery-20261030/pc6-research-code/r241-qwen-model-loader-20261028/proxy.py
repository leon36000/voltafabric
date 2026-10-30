#!/usr/bin/python3
import os,socket,sys,threading
path='/run/media/pc1/Infrastructure/Naos/.r241-session.sock';s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(path)
def incoming():
 try:
  while True:
   data=os.read(0,65536)
   if not data:break
   s.sendall(data)
 finally:
  try:s.shutdown(socket.SHUT_WR)
  except OSError:pass
threading.Thread(target=incoming,daemon=True).start()
while True:
 data=s.recv(65536)
 if not data:break
 os.write(1,data)
s.close()
