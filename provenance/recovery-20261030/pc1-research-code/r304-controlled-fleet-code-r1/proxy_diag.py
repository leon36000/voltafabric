"""Run the existing local proxy and retain bounded diagnostic stderr privately."""
import os,runpy,sys
if len(sys.argv)!=4:raise ValueError('proxy, socket, diagnostic path required')
proxy,sock,log=sys.argv[1:]
fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
os.dup2(fd,2);os.close(fd)
sys.argv=[proxy,sock]
runpy.run_path(proxy,run_name='__main__')
