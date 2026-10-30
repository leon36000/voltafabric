#include "pipe_client.hpp"
#include "pool_hook.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
namespace {
void put32(std::vector<uint8_t>& b,uint32_t v){for(int i=0;i<4;++i)b.push_back(uint8_t(v>>(8*i)));}
void put64(std::vector<uint8_t>& b,uint64_t v){for(int i=0;i<8;++i)b.push_back(uint8_t(v>>(8*i)));}
bool readn(int fd,size_t n){std::vector<uint8_t> b(n);size_t at=0;while(at<n){auto z=read(fd,b.data()+at,n-at);if(z<=0)return false;at+=size_t(z);}return true;}
bool sendn(int fd,const std::vector<uint8_t>& b){size_t at=0;while(at<b.size()){auto z=write(fd,b.data()+at,b.size()-at);if(z<=0)return false;at+=size_t(z);}return true;}
int test(int mode){
    int fd[2];if(socketpair(AF_UNIX,SOCK_STREAM,0,fd)) return 1;
    pid_t pid=fork();if(pid<0)return 1;
    if(pid==0){close(fd[0]); if(!readn(fd[1],28+4*(2560+6)))_exit(10);
        std::vector<uint8_t> msg; msg.insert(msg.end(),{'V','F','R','2'});put32(msg,0);put64(msg,mode==3?43:42);put32(msg,2);put32(msg,2560);put32(msg,mode==5?7:0);
        for(int row=0;row<2;++row){put32(msg,(mode==1 && row==1)?0:row*2);for(int i=0;i<2560;++i){float x=float(row+1);uint32_t bits;std::memcpy(&bits,&x,4);if(mode==4 && row==1 && i==0)bits=0x7fc00000;put32(msg,bits);}}
        if(mode==2)msg.resize(msg.size()-16);
        if(!sendn(fd[1],msg))_exit(11);
        close(fd[1]);_exit(0);
    }
    close(fd[1]);vf::strata_bridge::Client client(fd[0],fd[0],2000,0);std::vector<float> x(2560,.125f),out(3*2560,9.f);std::vector<int32_t> ids{0,2,1},kinds{-1,0,-1};std::string error;
    bool ok=client.run(42,1,3,x,ids,kinds,out,error); close(fd[0]);int status=0;waitpid(pid,&status,0);
    if(mode==0){if(!ok)return 1;for(size_t i=0;i<2560;++i)if(out[i]!=1.f||out[2560+i]!=9.f||out[5120+i]!=2.f)return 1;}
    else if(ok || !std::all_of(out.begin(),out.end(),[](float v){return v==9.f;}))return 1;
    return WIFEXITED(status)&&WEXITSTATUS(status)==0?0:1;
}
int hook_failure() {
    int fd[2]; if(socketpair(AF_UNIX,SOCK_STREAM,0,fd)) return 1;
    pid_t pid=fork(); if(pid<0)return 1;
    if(pid==0) {
        close(fd[0]); if(!readn(fd[1],28+4*(2560+4)))_exit(10);
        std::vector<uint8_t> msg; msg.insert(msg.end(),{'V','F','R','2'});put32(msg,0);put64(msg,55);put32(msg,2);put32(msg,2560);put32(msg,0);
        for(int row=0;row<2;++row){put32(msg,0);for(int i=0;i<2560;++i)put32(msg,0x3f800000);}
        if(!sendn(fd[1],msg))_exit(11);
        close(fd[1]);_exit(0);
    }
    close(fd[1]); vf::strata_bridge::Client client(fd[0],fd[0],2000,0);
    vf::strata_bridge::PoolContext ctx{client,0,55,false,{}, {}};
    std::vector<float> x(2560,0.125f),out(2*2560,9.f); int32_t ids[2]={0,1};
    bool first=vf::strata_bridge::pool_cpu_only(&ctx,x.data(),ids,1,2,out.data(),0);
    bool second=vf::strata_bridge::pool_cpu_only(&ctx,x.data(),ids,1,2,out.data(),0);
    close(fd[0]); int status=0; waitpid(pid,&status,0);
    return first || second || !ctx.terminal || !std::all_of(out.begin(),out.end(),[](float v){return v==9.f;}) || !WIFEXITED(status) || WEXITSTATUS(status)!=0;
}
}
int main(){for(int mode=0;mode<6;++mode)if(test(mode)){std::fprintf(stderr,"pipe client test failed %d\n",mode);return 1;}return hook_failure();}
