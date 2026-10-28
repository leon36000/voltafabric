import concurrent.futures
import http.server
import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys
import urllib.error
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from vf_agents import Gateway,Ledger,RequestFailure,parse_proposal,safe_context,validate_queue


def task(name='t1',write_set=None):
    return {'id':name,'model_key':'reviewer','objective':'Inspect one contract.',
            'files':['src/a.rs'],'write_set':[] if write_set is None else write_set}

def proposal():
    return {'summary':'A bounded proposal.','findings':[], 'patch':'','commands_to_run':['cargo test --locked'], 'verification':'NOT_EXECUTED'}

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'src').mkdir();(self.root/'src/a.rs').write_text('fn main() {}\n')
    def tearDown(self):self.tmp.cleanup()
    def test_exact_allowlisted_file(self):self.assertEqual(safe_context(self.root,['src/a.rs']),{'src/a.rs':'fn main() {}\n'})
    def test_parent_escape(self):
        with self.assertRaises(ValueError):safe_context(self.root,['../x'])
    def test_absolute_path(self):
        with self.assertRaises(ValueError):safe_context(self.root,[str(self.root/'src/a.rs')])
    def test_secret_filename(self):
        (self.root/'.env').write_text('TOKEN=secret')
        with self.assertRaises(ValueError):safe_context(self.root,['.env'])
    def test_secret_names_with_allowed_extensions_are_excluded(self):
        # .json is an allowed extension, so only the private-path filter can reject
        # these. The existing .env case is also caught by the extension rule and so
        # cannot distinguish the filter being present from it being absent.
        for name in ('credentials.json','my_secret.json','.env.json'):
            (self.root/name).write_text('{}')
            with self.assertRaises(ValueError,msg=name):safe_context(self.root,[name])
    def test_symlink_escape(self):
        (self.root/'src/link.rs').symlink_to('/etc/passwd')
        with self.assertRaises(ValueError):safe_context(self.root,['src/link.rs'])
    def test_total_byte_budget(self):
        with self.assertRaises(ValueError):safe_context(self.root,['src/a.rs'],max_bytes=4)
    def test_duplicate_context(self):
        with self.assertRaises(ValueError):safe_context(self.root,['src/a.rs','src/a.rs'])
    def test_private_artifacts_excluded(self):
        (self.root/'.runs').mkdir();(self.root/'.runs/a.json').write_text('{}')
        with self.assertRaises(ValueError):safe_context(self.root,['.runs/a.json'])
    def test_binary_rejected(self):
        (self.root/'src/b.rs').write_bytes(b'\xff\x00')
        with self.assertRaises(ValueError):safe_context(self.root,['src/b.rs'])

class QueueTests(unittest.TestCase):
    def test_valid_queue(self):self.assertEqual(validate_queue([task()]),[task()])
    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):validate_queue([task(),task()])
    def test_traversal_id(self):
        with self.assertRaises(ValueError):validate_queue([task('../x')])
    def test_conflicting_writers(self):
        with self.assertRaises(ValueError):validate_queue([task('a',['src/x.rs']),task('b',['src/x.rs'])])
    def test_unknown_task_fields(self):
        t=task();t['shell']='do something'
        with self.assertRaises(ValueError):validate_queue([t])
    def test_empty_objective(self):
        t=task();t['objective']=''
        with self.assertRaises(ValueError):validate_queue([t])
    def test_contextual_reviewers_can_overlap(self):self.assertEqual(len(validate_queue([task('a'),task('b')])),2)

class ProposalTests(unittest.TestCase):
    def test_valid(self):self.assertEqual(parse_proposal(proposal()),proposal())
    def test_cannot_claim_tests_run(self):
        p=proposal();p['verification']='PASSED'
        with self.assertRaises(ValueError):parse_proposal(p)
    def test_unknown_fields(self):
        p=proposal();p['tool_calls']=['shell']
        with self.assertRaises(ValueError):parse_proposal(p)
    def test_patch_is_only_data(self):
        p=proposal();p['patch']='UNTRUSTED shell-looking text; never executed'
        self.assertEqual(parse_proposal(p)['patch'],p['patch'])
    def test_bad_finding(self):
        p=proposal();p['findings']=[{'severity':'great','path':'../secret','line':0,'claim':'x'}]
        with self.assertRaises(ValueError):parse_proposal(p)

class LedgerTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'ledger.sqlite'
    def tearDown(self):self.tmp.cleanup()
    def test_reserve_and_complete(self):
        l=Ledger(self.path);self.assertTrue(l.reserve('a','hash',128));l.sent('a');l.finish('a','completed',{'prompt_tokens':10,'completion_tokens':8})
        self.assertEqual(l.rows()[0]['state'],'completed')
    def test_duplicate_never_resubmitted(self):
        l=Ledger(self.path);l.reserve('a','hash',128)
        with self.assertRaises(ValueError):l.reserve('a','hash',128)
    def test_changed_content_same_id_conflicts(self):
        l=Ledger(self.path);l.reserve('a','hash',128)
        with self.assertRaises(ValueError):l.reserve('a','other',128)
    def test_atomic_request_budget(self):
        l=Ledger(self.path,max_requests=5,max_output_tokens=10000)
        def reserve(i):
            try:l.reserve(str(i),'hash',100);return True
            except ValueError:return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:ok=list(pool.map(reserve,range(30)))
        self.assertEqual(sum(ok),5);self.assertEqual(len(l.rows()),5)
    def test_output_reservation_budget(self):
        l=Ledger(self.path,max_requests=10,max_output_tokens=200);l.reserve('a','h',128)
        with self.assertRaises(ValueError):l.reserve('b','h',128)
    def test_unknown_keeps_reservation(self):
        l=Ledger(self.path,max_requests=1);l.reserve('a','h',100);l.sent('a');l.finish('a','unknown')
        with self.assertRaises(ValueError):l.reserve('b','h',100)
        self.assertEqual(l.rows()[0]['state'],'unknown')
    def test_finish_requires_sent(self):
        l=Ledger(self.path);l.reserve('a','h',100)
        with self.assertRaises(ValueError):l.finish('a','completed')
    def test_limits_cannot_silently_change(self):
        Ledger(self.path,max_requests=3)
        with self.assertRaises(ValueError):Ledger(self.path,max_requests=30)

class Handler(http.server.BaseHTTPRequestHandler):
    mode='ok';seen=[]
    def log_message(self,*args):pass
    def do_GET(self):
        self.__class__.seen.append((self.path,self.headers.get('Authorization')))
        if self.mode=='redirect':
            self.send_response(302);self.send_header('Location','/steal');self.end_headers();return
        self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
        self.wfile.write(json.dumps({'data':[{'id':'proxy/exact-alias'}]}).encode())
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.__class__.seen.append(data)
        if self.mode=='rate':self.send_response(429);self.end_headers();return
        if self.mode=='server':self.send_response(500);self.end_headers();return
        self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
        self.wfile.write(json.dumps({'model':'upstream-actual','choices':[{'message':{'content':json.dumps(proposal())}}],
                                    'usage':{'prompt_tokens':20,'completion_tokens':30}}).encode())

class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def setUp(self):Handler.mode='ok';Handler.seen=[]
    def gateway(self):return Gateway(self.url,'dummy-test-key',allow_local_http=True)
    def test_discovery_exact_ids(self):self.assertEqual(self.gateway().models(),['proxy/exact-alias'])
    def test_auth_not_in_url(self):
        self.gateway().models();path,key=Handler.seen[0];self.assertNotIn('dummy',path);self.assertEqual(key,'Bearer dummy-test-key')
    def test_completion_is_proposal_not_execution(self):
        r=self.gateway().propose('proxy/exact-alias',[{'role':'user','content':'test'}],128)
        self.assertEqual(r['proposal']['verification'],'NOT_EXECUTED');self.assertEqual(r['usage']['completion_tokens'],30)
    def test_no_redirect_with_credentials(self):
        Handler.mode='redirect'
        with self.assertRaises(RequestFailure):self.gateway().models()
        self.assertEqual(len(Handler.seen),1)
    def test_rate_limit_not_retried(self):
        Handler.mode='rate'
        with self.assertRaises(RequestFailure) as c:self.gateway().propose('m',[],128)
        self.assertFalse(c.exception.uncertain);self.assertEqual(len(Handler.seen),1)
    def test_500_is_uncertain(self):
        Handler.mode='server'
        with self.assertRaises(RequestFailure) as c:self.gateway().propose('m',[],128)
        self.assertTrue(c.exception.uncertain)
    def test_unknown_transport_outcome_stays_uncertain(self):
        gateway=self.gateway()
        class Failing:
            def open(self,*a,**k):raise urllib.error.URLError('synthetic transport failure')
        gateway.opener=Failing()
        with self.assertRaises(RequestFailure) as c:gateway.propose('proxy/exact-alias',[{'role':'user','content':'x'}],128)
        self.assertIn('TRANSPORT_OUTCOME_UNKNOWN',str(c.exception))
        self.assertTrue(c.exception.uncertain,'an unknown billing outcome must never be reported as certain')
    def test_direct_provider_blocked(self):
        with self.assertRaises(ValueError):Gateway('https://openrouter.ai/api','key')
    def test_plain_http_requires_explicit_permission(self):
        with self.assertRaises(ValueError):Gateway(self.url,'key')
    def test_url_credentials_rejected(self):
        with self.assertRaises(ValueError):Gateway('https://key@proxy.example','key')
    def test_query_secret_rejected(self):
        with self.assertRaises(ValueError):Gateway('https://proxy.example/?key=x','key')

if __name__=='__main__':unittest.main()

class RunnerCliTests(unittest.TestCase):
    def setUp(self):
        import os
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'src').mkdir();(self.root/'src/a.rs').write_text('fn a() {}')
        self.cfg={'aliases':{'reviewer':'proxy/exact-alias'},'workers':2,'max_requests':4,
                  'max_reserved_output_tokens':1024,'max_tokens':128,'context_byte_limit':1024}
        self.config=self.root/'config.json';self.queue=self.root/'queue.json'
        self.config.write_text(json.dumps(self.cfg));self.queue.write_text(json.dumps([task()]))
        self.script=Path(__file__).resolve().parents[2]/'scripts/vf_agents.py'
        self.env=os.environ.copy();self.env.pop('VF_LITELLM_URL',None);self.env.pop('LITELLM_API_KEY',None)
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        Handler.mode='ok';Handler.seen=[]
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()
    def run_cli(self,extra=()):
        import subprocess
        return subprocess.run([sys.executable,str(self.script),'run','--root',str(self.root),'--config',str(self.config),
              '--queue',str(self.queue),*extra],env=self.env,capture_output=True,text=True,timeout=10)
    def live(self):
        self.env['VF_LITELLM_URL']=f'http://127.0.0.1:{self.server.server_port}';self.env['LITELLM_API_KEY']='dummy-test-key'
        return ['--execute','--authorized-spend','--allow-local-http']
    def test_dry_run_needs_no_network_or_key(self):
        r=self.run_cli();self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['mode'],'DRY_RUN_NO_NETWORK');self.assertEqual(Handler.seen,[])
    def test_paid_calls_require_authorization(self):
        r=self.run_cli(['--execute']);self.assertNotEqual(r.returncode,0);self.assertEqual(Handler.seen,[])
    def test_authorization_is_required_even_with_a_reachable_gateway(self):
        self.live()
        r=self.run_cli(['--execute','--allow-local-http'])
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(Handler.seen,[],'nothing may reach the gateway without --authorized-spend')
    def test_live_mock_and_replay_block(self):
        self.assertEqual(self.run_cli(self.live()).returncode,0)
        self.assertEqual(len([x for x in Handler.seen if isinstance(x,dict)]),1)
        again=self.run_cli(self.live());self.assertNotEqual(again.returncode,0)
        self.assertEqual(len([x for x in Handler.seen if isinstance(x,dict)]),1)
        text=(self.root/'.runs/first-wave/t1.proposal.json').read_text();self.assertNotIn('dummy-test-key',text)
    def test_missing_alias_never_posts(self):
        self.cfg['aliases']['reviewer']='invented';self.config.write_text(json.dumps(self.cfg))
        r=self.run_cli(self.live());self.assertNotEqual(r.returncode,0);self.assertEqual(len([x for x in Handler.seen if isinstance(x,dict)]),0)
    def test_run_symlink_checked_before_creation(self):
        with tempfile.TemporaryDirectory() as d:
            outside=Path(d);(self.root/'.runs').symlink_to(outside,target_is_directory=True)
            r=self.run_cli(self.live());self.assertNotEqual(r.returncode,0)
            self.assertFalse((outside/'first-wave').exists())
    def test_workers_bound_before_network(self):
        self.cfg['workers']=999;self.config.write_text(json.dumps(self.cfg))
        r=self.run_cli(self.live());self.assertNotEqual(r.returncode,0);self.assertEqual(Handler.seen,[])
