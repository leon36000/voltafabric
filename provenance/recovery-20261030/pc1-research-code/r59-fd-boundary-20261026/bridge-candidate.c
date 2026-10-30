/* VoltaFabric laboratory only. Native rsockets ABI adapter; no engine logic. */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <poll.h>
#include <time.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

static int (*api_socket)(int,int,int);
static int (*api_bind)(int,const struct sockaddr *,socklen_t);
static int (*api_listen)(int,int);
static int (*api_accept)(int,struct sockaddr *,socklen_t *);
static int (*api_connect)(int,const struct sockaddr *,socklen_t);
static int (*api_close)(int);
static ssize_t (*api_send)(int,const void *,size_t,int);
static ssize_t (*api_recv)(int,void *,size_t,int);
static int (*api_opt)(int,int,int,const void *,socklen_t);
static int (*api_shutdown)(int,int);
static int (*api_poll)(struct pollfd *,nfds_t,int);
static int (*api_fcntl)(int,int,...);
static int (*api_peer)(int,struct sockaddr *,socklen_t *);

static void require(int pass, const char *why) {
    if (!pass) { fprintf(stderr,"R24_FAIL %s errno=%d\n",why,errno); exit(2); }
}
static void load(void) {
    void *lib=dlopen("librdmacm.so.1", RTLD_NOW|RTLD_LOCAL);
    require(lib!=NULL,"dlopen");
#define SYMBOL(slot,name) do { void *p=dlsym(lib,name); require(p!=NULL,name); _Static_assert(sizeof(slot)==sizeof(p),"Linux ABI pointer size"); memcpy(&slot,&p,sizeof(p)); } while(0)
    SYMBOL(api_shutdown,"rshutdown"); SYMBOL(api_poll,"rpoll"); SYMBOL(api_fcntl,"rfcntl"); SYMBOL(api_socket,"rsocket"); SYMBOL(api_bind,"rbind");
    SYMBOL(api_listen,"rlisten"); SYMBOL(api_accept,"raccept");
    SYMBOL(api_connect,"rconnect"); SYMBOL(api_close,"rclose");
    SYMBOL(api_send,"rsend"); SYMBOL(api_recv,"rrecv");
    SYMBOL(api_opt,"rsetsockopt"); SYMBOL(api_peer,"rgetpeername");
}
static int transfer(int fd, void *buffer, size_t len, int sending) {
    size_t done=0;
    while(done<len) {
        ssize_t n=sending?api_send(fd,(char *)buffer+done,len-done,0):api_recv(fd,(char *)buffer+done,len-done,0);
        if(n<0 && errno==EINTR) continue;
        if(n<=0) return -1;
        require((size_t)n<=len-done,"transfer size"); done+=(size_t)n;
    }
    return 0;
}
static void cookie(const char *path,unsigned char out[64]) {
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    require(fd>=0,"cookie open"); struct stat st;
    require(fstat(fd,&st)==0 && S_ISREG(st.st_mode) && st.st_uid==getuid() && (st.st_mode&077)==0 && st.st_size==64,"cookie ownership");
    size_t got=0;
    while(got<64) { ssize_t n=read(fd,out+got,64-got); if(n<0 && errno==EINTR)continue; require(n>0,"cookie read");got+=(size_t)n; }
    close(fd);
}
static int equal(const unsigned char *a,const unsigned char *b,size_t n) {
    unsigned char v=0; for(size_t i=0;i<n;i++) v|=a[i]^b[i]; return v==0;
}

/* Two fixed-size queues, no application framing or inference semantics here. */
static int bridge(int fd,int source,int dest,int server_side) {
    unsigned char tx[65536],rx[65536];
    size_t tx_len=0,tx_off=0,rx_len=0,rx_off=0;
    int input_eof=0,net_eof=0,half_closed=0,dest_closed=0;
    uint64_t sent=0,received=0;
    require(api_fcntl(fd,F_SETFL,O_NONBLOCK)==0,"rsocket nonblocking");
    int f=fcntl(source,F_GETFL);require(f>=0 && fcntl(source,F_SETFL,f|O_NONBLOCK)==0,"source nonblocking");
    f=fcntl(dest,F_GETFL);require(f>=0 && fcntl(dest,F_SETFL,f|O_NONBLOCK)==0,"dest nonblocking");
    for(;;) {
        if(input_eof && tx_off==tx_len && !half_closed) {
            require(api_shutdown(fd,SHUT_WR)==0,"RDMA write shutdown");half_closed=1;
        }
        if(net_eof && rx_off==rx_len) {
            if(!server_side) {
                require(tx_off==tx_len,"peer closed with unsent data UNKNOWN");break;
            }
            /* Peer SHUT_WR ends only worker stdin. Drain worker stdout separately. */
            if(!dest_closed) {
                require(close(dest)==0,"worker stdin EOF");dest=-1;dest_closed=1;
            }
            if(input_eof && tx_off==tx_len && half_closed)break;
        }
        struct pollfd p[3]={
            {.fd=fd,.events=(short)((!net_eof && rx_off==rx_len?POLLIN:0)|(tx_off<tx_len?POLLOUT:0))},
            {.fd=(!input_eof && tx_off==tx_len)?source:-1,.events=POLLIN},
            {.fd=rx_off<rx_len?dest:-1,.events=POLLOUT}
        };
        /* Do not block the rsockets progress path while local worker pipes may wake. */
        int n=api_poll(p,3,0);
        if(n==0) {
            struct pollfd local[2]={p[1],p[2]};
            const struct timespec wait={.tv_sec=0,.tv_nsec=100000};
            n=ppoll(local,2,&wait,NULL);
            p[1].revents=local[0].revents;p[2].revents=local[1].revents;
        }
        if(n<0 && errno==EINTR)continue;
        require(n>=0,"rpoll");
        for(unsigned i=0;i<3;i++)require(!(p[i].revents&POLLNVAL),"invalid poll descriptor");
        if(!input_eof && tx_off==tx_len && (p[1].revents&(POLLIN|POLLHUP|POLLERR))) {
            ssize_t got=read(source,tx,sizeof(tx));
            if(got==0)input_eof=1;
            else if(got>0){tx_off=0;tx_len=(size_t)got;}
            else require(errno==EINTR || errno==EAGAIN,"source read");
        }
        if(tx_off<tx_len && (p[0].revents&POLLOUT)) {
            ssize_t wrote=api_send(fd,tx+tx_off,tx_len-tx_off,MSG_DONTWAIT);
            if(wrote>0){require((size_t)wrote<=tx_len-tx_off,"send count");tx_off+=(size_t)wrote;sent+=(uint64_t)wrote;}
            else require(wrote<0 && (errno==EINTR || errno==EAGAIN),"RDMA send UNKNOWN");
        }
        if(!net_eof && rx_off==rx_len && (p[0].revents&(POLLIN|POLLHUP|POLLERR))) {
            ssize_t got=api_recv(fd,rx,sizeof(rx),MSG_DONTWAIT);
            if(got==0)net_eof=1;
            else if(got>0){rx_off=0;rx_len=(size_t)got;received+=(uint64_t)got;}
            else require(errno==EINTR || errno==EAGAIN,"RDMA receive UNKNOWN");
        }
        if(rx_off<rx_len && (p[2].revents&(POLLOUT|POLLHUP|POLLERR))) {
            ssize_t wrote=write(dest,rx+rx_off,rx_len-rx_off);
            if(wrote>0){require((size_t)wrote<=rx_len-rx_off,"write count");rx_off+=(size_t)wrote;}
            else require(wrote<0 && (errno==EINTR || errno==EAGAIN),"destination write UNKNOWN");
        }
    }
    fprintf(stderr,"R24_STREAM_BYTES sent=%llu received=%llu\n",(unsigned long long)sent,(unsigned long long)received);
    return dest_closed;
}
int main(int argc,char **argv) {
    require(argc==6,"usage role bind-ip port cookie worker");
    int server=!strcmp(argv[1],"server");require(server || !strcmp(argv[1],"client"),"role");
    require(!strcmp(argv[2],"10.40.13.1") || !strcmp(argv[2],"10.40.13.2"),"private IP");
    char *end;long port=strtol(argv[3],&end,10);require(*end=='\0' && port>=18900 && port<=18919,"port");
    unsigned char keys[64],auth[32];cookie(argv[4],keys);
    if(server){struct stat st;require(argv[5][0]=='/' && stat(argv[5],&st)==0 && S_ISREG(st.st_mode) && st.st_uid==getuid() && access(argv[5],X_OK)==0,"trusted local worker path");}
    signal(SIGPIPE,SIG_IGN);alarm(12);load();
    int fd=api_socket(AF_INET,SOCK_STREAM,0);require(fd>=0,"rsocket");int cap=65536;
    require(api_opt(fd,SOL_SOCKET,SO_RCVBUF,&cap,sizeof(cap))==0,"receive cap");
    require(api_opt(fd,SOL_SOCKET,SO_SNDBUF,&cap,sizeof(cap))==0,"send cap");
    struct sockaddr_in own={.sin_family=AF_INET,.sin_port=htons(server?(uint16_t)port:0)};inet_pton(AF_INET,argv[2],&own.sin_addr);
    require(api_bind(fd,(struct sockaddr *)&own,sizeof(own))==0,"bind");
    const char *peerip=!strcmp(argv[2],"10.40.13.1")?"10.40.13.2":"10.40.13.1";
    if(server){require(api_listen(fd,1)==0,"listen");fprintf(stderr,"R24_LISTENING\n");int next=api_accept(fd,NULL,NULL);require(next>=0,"accept");api_close(fd);fd=next;}
    else {struct sockaddr_in peer={.sin_family=AF_INET,.sin_port=htons((uint16_t)port)};inet_pton(AF_INET,peerip,&peer.sin_addr);require(api_connect(fd,(struct sockaddr *)&peer,sizeof(peer))==0,"connect");}
    struct sockaddr_in actual;struct in_addr expected; socklen_t len=sizeof(actual);inet_pton(AF_INET,peerip,&expected);
    require(api_peer(fd,(struct sockaddr *)&actual,&len)==0 && actual.sin_family==AF_INET && actual.sin_addr.s_addr==expected.s_addr,"peer IP");
    if(server){int authorized=transfer(fd,auth,32,0)==0 && equal(auth,keys,32);if(!authorized){api_close(fd);require(0,"client cookie");}require(transfer(fd,keys+32,32,1)==0,"server cookie");}
    else {require(transfer(fd,keys,32,1)==0,"client cookie");require(transfer(fd,auth,32,0)==0 && equal(auth,keys+32,32),"server cookie");}
    fprintf(stderr,"R24_AUTHENTICATED\n");
    int status=0;
    if(server) {
        int input[2],output[2];require(pipe2(input,O_CLOEXEC)==0 && pipe2(output,O_CLOEXEC)==0,"worker pipes");
        pid_t parent=getpid(),pid=fork();require(pid>=0,"fork worker");
        if(pid==0){
            if(prctl(PR_SET_PDEATHSIG,SIGKILL)!=0 || getppid()!=parent)_exit(125);
            if(dup2(input[0],STDIN_FILENO)<0 || dup2(output[1],STDOUT_FILENO)<0)_exit(126);
            close(input[0]);close(input[1]);close(output[0]);close(output[1]);
            /* close the native rsocket in the child without invoking fork-unsafe RDMA APIs. */
            if(close_range(3,~0U,0)!=0)_exit(126);
            execl(argv[5],argv[5],(char *)NULL);_exit(127);
        }
        fprintf(stderr,"R24_WORKER_PID %ld\n",(long)pid);close(input[0]);close(output[1]);
        int dest_closed=bridge(fd,output[0],input[1],1);if(!dest_closed)close(input[1]);close(output[0]);api_close(fd);
        require(waitpid(pid,&status,0)==pid,"wait worker");
        require(WIFEXITED(status) && WEXITSTATUS(status)==0,"worker exit UNKNOWN");
        fprintf(stderr,"R24_WORKER_EXIT 0\n");
    } else {bridge(fd,STDIN_FILENO,STDOUT_FILENO,0);api_close(fd);}
    return 0;
}
