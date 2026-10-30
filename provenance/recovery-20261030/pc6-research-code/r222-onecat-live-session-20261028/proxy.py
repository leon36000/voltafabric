#!/usr/bin/python3
import os,socket,sys,threading
path='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r222-session-r1.sock';s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(path)
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
