#!/usr/bin/env python3
import json,sys,time
sys.stdin.readline()
print(json.dumps({'ok':True,'state':'ready'}),flush=True)
sys.stdin.readline()
time.sleep(1.0)
