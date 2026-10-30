"""Actual fork/exec FD boundary from the bridge; no rsocket, NIC or GPU calls."""

import errno
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "scripts/rdma_cpu_bridge.c"
WORK = ROOT / ".work/rdma-fd-boundary-tests"
KEEP = False
PROFILES = [("debug", ["-O0", "-g"]), ("release", ["-O2"])]
PAYLOAD = bytes([0, 255, 10, 13, 0, 127, 128]) + b"FD_BOUNDARY_EXACT_BYTES"
FIXTURE = r"""#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
int main(void) {
    const char *p=getenv("R59_TEST_FD"); if(!p)return 91;
    int open=fcntl(atoi(p),F_GETFD)>=0; unsigned char b[128];size_t n=0;unsigned long sum=0;ssize_t got;
    while((got=read(0,b,sizeof(b)))>0){n+=(size_t)got;for(ssize_t i=0;i<got;i++)sum+=b[i];}
    if(got<0)return 92;
    printf("{\"executed\":true,\"extra_fd_open\":%s,\"input_bytes\":%zu,\"checksum\":%lu}\n",open?"true":"false",n,sum);
    return 0;
}
"""
HARNESS = r"""#define _GNU_SOURCE
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
static int failure;
extern int __real_close_range(unsigned int,unsigned int,int);
int __wrap_close_range(unsigned int first,unsigned int last,int flags) {
    dprintf(2,"R59_CLOSE_RANGE first=%u last=%u flags=%d injected=%d\n",first,last,flags,failure);
    if(failure){errno=failure;return -1;}
    return __real_close_range(first,last,flags);
}
int main(int argc,char **args) {
    if(argc!=3)return 89;
    failure=atoi(args[2]);signal(SIGPIPE,SIG_IGN);
    int input[2],output[2];if(pipe2(input,O_CLOEXEC)||pipe2(output,O_CLOEXEC))return 88;
    int original=open("/dev/null",O_RDONLY);if(original<0)return 87;
    int extra=fcntl(original,F_DUPFD,99);close(original);if(extra<99)return 86;
    char fd[32];snprintf(fd,sizeof(fd),"%d",extra);if(setenv("R59_TEST_FD",fd,1))return 85;
    unsigned char b[4096];size_t n=0;ssize_t got;
    while(n<sizeof(b)&&(got=read(0,b+n,sizeof(b)-n))>0)n+=(size_t)got;
    char *values[6]={NULL,NULL,NULL,NULL,NULL,args[1]};char **argv=values;
    pid_t parent=getpid(),pid=fork();if(pid<0)return 84;
    /* EXACT_CHILD_BLOCK */
    close(input[0]);close(output[1]);close(extra);
    ssize_t sent=write(input[1],b,n);if(sent<0&&errno!=EPIPE)return 83;
    if(sent>=0&&(size_t)sent!=n)return 82;
    close(input[1]);
    while((got=read(output[0],b,sizeof(b)))>0){if(write(1,b,(size_t)got)!=got)return 81;}
    close(output[0]);int status;if(waitpid(pid,&status,0)!=pid)return 80;
    if(!WIFEXITED(status))return 79;
    dprintf(2,"R59_CHILD_EXIT %d\n",WEXITSTATUS(status));return WEXITSTATUS(status);
}
"""


def child_block(text):
    start = text.index("        if(pid==0){")
    end = text.index('        fprintf(stderr,"R24_WORKER_PID', start)
    block = text[start:end]
    if block.count("close_range(3,~0U,0)") != 1 or block.count("execl(argv[5]") != 1:
        raise ValueError("unexpected bridge fork/exec boundary")
    return block


class WorkerFdBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=WORK)
        cls.path = Path(cls.temp.name)
        if KEEP:
            cls.temp._finalizer.detach()
        cls.source = BRIDGE.read_bytes()
        cls.block = child_block(cls.source.decode())
        (cls.path / "bridge-source.c").write_bytes(cls.source)
        (cls.path / "child-block.c").write_text(cls.block)
        (cls.path / "fixture.c").write_text(FIXTURE)
        (cls.path / "harness.c").write_text(
            HARNESS.replace("/* EXACT_CHILD_BLOCK */", cls.block)
        )
        cls.binaries = []
        cls.results = []
        cc = shutil.which("cc")
        if not cc:
            raise RuntimeError("C compiler required for actual FD regression")
        for name, flags in PROFILES:
            dest = cls.path / name
            dest.mkdir()
            for src, binary, extra in [
                ("fixture.c", "worker", []),
                ("harness.c", "harness", ["-Wl,--wrap=close_range"]),
            ]:
                argv = [
                    cc,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *flags,
                    str(cls.path / src),
                    *extra,
                    "-o",
                    str(dest / binary),
                ]
                q = subprocess.run(argv, capture_output=True, text=True, timeout=30)
                (dest / (binary + ".compile.json")).write_text(
                    json.dumps(
                        {
                            "argv": argv,
                            "exit_code": q.returncode,
                            "stdout": q.stdout,
                            "stderr": q.stderr,
                        },
                        indent=2,
                    )
                )
                if q.returncode:
                    raise RuntimeError(q.stderr)
            cls.binaries.append((name, dest / "harness", dest / "worker"))

    @classmethod
    def tearDownClass(cls):
        if KEEP:
            cls.temp._finalizer.detach()
        else:
            cls.temp.cleanup()

    def invoke(self, mode, harness, worker, error):
        q = subprocess.run(
            [str(harness), str(worker), str(error)],
            input=PAYLOAD,
            capture_output=True,
            timeout=5,
        )
        result = dict(
            mode=mode,
            argv=q.args,
            exit_code=q.returncode,
            stdout_hex=q.stdout.hex(),
            stderr=q.stderr.decode(),
            payload_hex=PAYLOAD.hex(),
            source_sha256=hashlib.sha256(self.source).hexdigest(),
            child_block_sha256=hashlib.sha256(self.block.encode()).hexdigest(),
        )
        self.results.append(result)
        (self.path / "executions.json").write_text(json.dumps(self.results, indent=2))
        self.assertEqual(q.stderr.count(b"R59_CLOSE_RANGE "), 1, q.stderr)
        self.assertIn(b"first=3 last=4294967295 flags=0", q.stderr)
        self.assertNotIn(b"ERROR: AddressSanitizer", q.stderr)
        self.assertNotIn(b"runtime error:", q.stderr)
        return q

    def test_real_close_then_exec_keeps_input_exact(self):
        for mode, harness, worker in self.binaries:
            with self.subTest(mode=mode):
                q = self.invoke(mode, harness, worker, 0)
                self.assertEqual(q.returncode, 0, q.stderr)
                self.assertEqual(
                    json.loads(q.stdout),
                    dict(
                        executed=True,
                        extra_fd_open=False,
                        input_bytes=len(PAYLOAD),
                        checksum=sum(PAYLOAD),
                    ),
                )

    def test_close_failure_cannot_execute_worker(self):
        for mode, harness, worker in self.binaries:
            for code in [errno.ENOSYS, errno.EPERM, errno.EINTR]:
                with self.subTest(mode=mode, error=code):
                    q = self.invoke(mode, harness, worker, code)
                    self.assertEqual(q.returncode, 126, (q.stdout, q.stderr))
                    self.assertEqual(
                        q.stdout, b"", "worker was executed after failed FD close"
                    )

    def test_exec_failure_is_distinct_after_successful_close(self):
        for mode, harness, worker in self.binaries:
            with self.subTest(mode=mode):
                q = self.invoke(mode, harness, worker.with_name("absent"), 0)
                self.assertEqual(q.returncode, 127, q.stderr)
                self.assertEqual(q.stdout, b"")


if __name__ == "__main__":
    unittest.main()
