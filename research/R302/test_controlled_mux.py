import os,pathlib,socket,tempfile,threading,time,unittest
from adapters.cpu_multiplex import serve_fleet, MultiplexError

class Tests(unittest.TestCase):
 def setup_fleet(self, serve, limit=4):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.sockpath=str(pathlib.Path(self.tmp.name)/'mux.sock');self.server=socket.socket(socket.AF_UNIX);self.server.bind(self.sockpath);self.server.listen(limit);self.addCleanup(self.server.close)
  self.rd,self.wr=os.pipe();self.addCleanup(os.close,self.rd);self.addCleanup(os.close,self.wr)
  self.result=[]
  def run():
   try:self.result.append(serve_fleet(self.server,self.rd,serve,limit,1))
   except BaseException as e:self.result.append(e)
  self.thread=threading.Thread(target=run);self.thread.start()
 def test_partial_stop_joins_real_connections(self):
  ready=threading.Event()
  def serve(c):
   ready.set(); self.assertEqual(c.recv(1),b'x'); c.sendall(b'y')
  self.setup_fleet(serve); c=socket.socket(socket.AF_UNIX);c.connect(self.sockpath);self.assertTrue(ready.wait(2));c.sendall(b'x');self.assertEqual(c.recv(1),b'y');c.close();os.write(self.wr,b'S');self.thread.join(2);self.assertFalse(self.thread.is_alive());self.assertEqual(self.result,[1])
 def test_stop_does_not_close_inflight(self):
  ready=threading.Event();release=threading.Event()
  def serve(c):ready.set();release.wait(2);c.sendall(b'z')
  self.setup_fleet(serve);c=socket.socket(socket.AF_UNIX);c.connect(self.sockpath);self.assertTrue(ready.wait(2));os.write(self.wr,b'S');time.sleep(.05);self.assertTrue(self.thread.is_alive());release.set();self.assertEqual(c.recv(1),b'z');c.close();self.thread.join(2);self.assertEqual(self.result,[1])
 def test_timeout_preserves_inflight_connection(self):
  ready=threading.Event();release=threading.Event();finished=threading.Event()
  def serve(c):
   ready.set();release.wait(5);c.sendall(b'z');finished.set()
  self.setup_fleet(serve);c=socket.socket(socket.AF_UNIX);c.connect(self.sockpath);self.assertTrue(ready.wait(2));os.write(self.wr,b'S');self.thread.join(2)
  self.assertIsInstance(self.result[0],MultiplexError);self.assertFalse(finished.is_set());release.set();self.assertEqual(c.recv(1),b'z');self.assertTrue(finished.wait(2));c.close()
 def test_invalid_control_fails_closed(self):
  self.setup_fleet(lambda c:None);os.write(self.wr,b'X');self.thread.join(2);self.assertIsInstance(self.result[0],MultiplexError)
 def test_normal_limit(self):
  self.setup_fleet(lambda c:c.sendall(b'a'),limit=1);c=socket.socket(socket.AF_UNIX);c.connect(self.sockpath);self.assertEqual(c.recv(1),b'a');c.close();self.thread.join(2);self.assertEqual(self.result,[1])
if __name__=='__main__':unittest.main()
