#include "pipe_client.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

namespace vf::strata_bridge {
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559, "bridge requires IEEE binary32");
namespace {
int64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void put32(std::vector<uint8_t>& b,uint32_t v) { for(int i=0;i<4;++i) b.push_back(uint8_t(v>>(8*i))); }
void put16(std::vector<uint8_t>& b,uint16_t v) { for(int i=0;i<2;++i) b.push_back(uint8_t(v>>(8*i))); }
void put64(std::vector<uint8_t>& b,uint64_t v) { for(int i=0;i<8;++i) b.push_back(uint8_t(v>>(8*i))); }
uint32_t get32(const uint8_t* p) { uint32_t v=0; for(int i=0;i<4;++i) v|=uint32_t(p[i])<<(8*i); return v; }
uint16_t get16(const uint8_t* p) { uint16_t v=0; for(int i=0;i<2;++i) v|=uint16_t(p[i])<<(8*i); return v; }
uint64_t get64(const uint8_t* p) { uint64_t v=0; for(int i=0;i<8;++i) v|=uint64_t(p[i])<<(8*i); return v; }
bool valid_identity_field(const std::string& value) {
    return !value.empty() && value.size() <= 256 && value.find('\0') == std::string::npos;
}
bool valid_identity(const SessionIdentity& identity) {
    return valid_identity_field(identity.run_id) && valid_identity_field(identity.model_revision) &&
           valid_identity_field(identity.plan_id);
}
void put_identity(std::vector<uint8_t>& out, const SessionIdentity& identity) {
    out.insert(out.end(), identity.run_id.begin(), identity.run_id.end());
    out.insert(out.end(), identity.model_revision.begin(), identity.model_revision.end());
    out.insert(out.end(), identity.plan_id.begin(), identity.plan_id.end());
}
bool identity_matches(const uint8_t* bytes, const SessionIdentity& identity) {
    const size_t run_size = identity.run_id.size(), model_size = identity.model_revision.size();
    return std::memcmp(bytes, identity.run_id.data(), run_size) == 0 &&
           std::memcmp(bytes + run_size, identity.model_revision.data(), model_size) == 0 &&
           std::memcmp(bytes + run_size + model_size, identity.plan_id.data(), identity.plan_id.size()) == 0;
}
bool local_socket(int fd) {
    sockaddr_storage address{}; socklen_t n=sizeof(address); int type=0; socklen_t t=sizeof(type);
    return getsockname(fd,reinterpret_cast<sockaddr*>(&address),&n)==0 && address.ss_family==AF_UNIX &&
           getsockopt(fd,SOL_SOCKET,SO_TYPE,&type,&t)==0 && type==SOCK_STREAM;
}
}
Client::Client(int in, int out, int timeout_ms, uint32_t layer)
    : read_fd_(in), write_fd_(out), timeout_ms_(timeout_ms), layer_(layer), identity_(), usable_(false) {
    initialize();
}
Client::Client(int in, int out, int timeout_ms, uint32_t layer, SessionIdentity identity)
    : read_fd_(in), write_fd_(out), timeout_ms_(timeout_ms), layer_(layer), identity_(std::move(identity)), usable_(false) {
    initialize();
}
void Client::initialize() noexcept {
    if (layer_>=48 || timeout_ms_<1 || timeout_ms_>60000 || !local_socket(read_fd_) || !local_socket(write_fd_)) return;
    if (identity_ && !valid_identity(*identity_)) return;
    int a=fcntl(read_fd_,F_GETFL), b=fcntl(write_fd_,F_GETFL);
    if (a<0 || b<0 || fcntl(read_fd_,F_SETFL,a|O_NONBLOCK)<0 || fcntl(write_fd_,F_SETFL,b|O_NONBLOCK)<0) return;
    usable_=true;
}
bool Client::transfer(bool writing,uint8_t* data,size_t size,int64_t deadline,std::string& error) {
    int fd=writing?write_fd_:read_fd_;
    while(size) {
        int64_t left=deadline-now_ms(); if(left<=0) { error="bridge deadline; outcome unknown"; return false; }
        pollfd p{fd,short(writing?POLLOUT:POLLIN),0};
        int n=poll(&p,1,int(std::min<int64_t>(left,60000)));
        if(n<0 && errno==EINTR) continue;
        if(n<=0 || (p.revents&(POLLERR|POLLNVAL)) || !(p.revents&(writing?POLLOUT:POLLIN))) { error="bridge socket unavailable; outcome unknown"; return false; }
        ssize_t done=writing?send(fd,data,size,MSG_NOSIGNAL):recv(fd,data,size,0);
        if(done<0 && (errno==EINTR || errno==EAGAIN || errno==EWOULDBLOCK)) continue;
        if(done<=0) { error="bridge socket closed; outcome unknown"; return false; }
        data+=done; size-=size_t(done);
    }
    return true;
}
bool Client::run(uint64_t batch,uint64_t attempt,int tokens,int k,const std::vector<float>& x,const std::vector<int32_t>& ids,
                 const std::vector<int32_t>& kinds,std::vector<float>& destination,std::string& error) try {
    error.clear();
    const bool bound=identity_.has_value();
    if(!usable_ || (bound && attempt==0) || tokens<1 || tokens>8 || k<1 || k>64 || tokens*k>128 || x.size()!=size_t(tokens)*2560 ||
       ids.size()!=size_t(tokens*k) || kinds.size()!=ids.size() || destination.size()!=ids.size()*2560 ||
       std::any_of(x.begin(),x.end(),[](float v){return !std::isfinite(v);}) ||
       std::any_of(kinds.begin(),kinds.end(),[](int32_t v){return v < -1 || v > 2;})) { error="invalid bridge input or terminal client"; return false; }
    std::vector<uint8_t> request;
    request.reserve((bound?52:28)+(bound?identity_->run_id.size()+identity_->model_revision.size()+identity_->plan_id.size():0)+4*(x.size()+ids.size()*2));
    if(bound) request.insert(request.end(),{'V','F','B','3'});
    else request.insert(request.end(),{'V','F','B','2'});
    put32(request,1); put64(request,batch); put32(request,uint32_t(tokens)); put32(request,uint32_t(k)); put32(request,layer_);
    if(bound) {
        put64(request,identity_->epoch);
        put16(request,uint16_t(identity_->run_id.size())); put16(request,uint16_t(identity_->model_revision.size()));
        put16(request,uint16_t(identity_->plan_id.size())); put16(request,0); put64(request,attempt);
        put_identity(request,*identity_);
    }
    for(float v:x) { uint32_t b; std::memcpy(&b,&v,4); put32(request,b); }
    for(int32_t v:ids) put32(request,uint32_t(v));
    for(int32_t v:kinds) put32(request,uint32_t(v));
    const int64_t deadline=now_ms()+timeout_ms_; uint8_t header[52]{};
    if(!transfer(true,request.data(),request.size(),deadline,error) || !transfer(false,header,bound?52:28,deadline,error)) { usable_=false; return false; }
    uint32_t status=get32(header+4), rows=get32(header+16), columns=get32(header+20);
    const size_t required=std::count(kinds.begin(),kinds.end(),-1);
    if(std::memcmp(header,bound?"VFR3":"VFR2",4)!=0 || status!=0 || get64(header+8)!=batch || rows!=required || columns!=2560 || get32(header+24)!=layer_) { usable_=false; error="invalid or refused bridge response"; return false; }
    if(bound) {
        const size_t id_size=identity_->run_id.size()+identity_->model_revision.size()+identity_->plan_id.size();
        if(get64(header+28)!=identity_->epoch || get16(header+36)!=identity_->run_id.size() ||
           get16(header+38)!=identity_->model_revision.size() || get16(header+40)!=identity_->plan_id.size() || get16(header+42)!=0 || get64(header+44)!=attempt) {
            usable_=false; error="bridge response identity header mismatch"; return false;
        }
        std::vector<uint8_t> echoed(id_size);
        if(!transfer(false,echoed.data(),echoed.size(),deadline,error)) { usable_=false; return false; }
        if(!identity_matches(echoed.data(),*identity_)) { usable_=false; error="bridge response identity mismatch"; return false; }
    }
    std::vector<uint8_t> response(size_t(rows)*(4+2560*4));
    if(!transfer(false,response.data(),response.size(),deadline,error)) { usable_=false; return false; }
    std::vector<std::pair<size_t,std::vector<float>>> staged; std::vector<bool> seen(ids.size(),false);
    for(size_t r=0;r<rows;++r) {
        size_t at=r*(4+2560*4); size_t pos=get32(response.data()+at);
        if(pos>=ids.size() || kinds[pos]!=-1 || seen[pos]) { usable_=false; error="invalid bridge slot coverage"; return false; }
        seen[pos]=true; std::vector<float> values(2560);
        for(size_t i=0;i<2560;++i) { uint32_t b=get32(response.data()+at+4+4*i); std::memcpy(&values[i],&b,4); if(!std::isfinite(values[i])) { usable_=false; error="non-finite bridge output"; return false; } }
        staged.emplace_back(pos,std::move(values));
    }
    for(const auto& row:staged) std::copy(row.second.begin(),row.second.end(),destination.begin()+row.first*2560);
    return true;
} catch (...) {
    usable_=false;
    error="bridge client failure";
    return false;
}
bool Client::shutdown(std::string& error) try {
    if(!usable_) { error="terminal bridge client"; return false; }
    const bool bound=identity_.has_value();
    std::vector<uint8_t> req;
    if(bound) req.insert(req.end(),{'V','F','B','3'});
    else req.insert(req.end(),{'V','F','B','2'});
    put32(req,2); put64(req,0); put32(req,0); put32(req,0); put32(req,layer_);
    if(bound) {
        put64(req,identity_->epoch);
        put16(req,uint16_t(identity_->run_id.size())); put16(req,uint16_t(identity_->model_revision.size()));
        put16(req,uint16_t(identity_->plan_id.size())); put16(req,0); put64(req,0);
        put_identity(req,*identity_);
    }
    uint8_t h[52]{}; int64_t deadline=now_ms()+timeout_ms_; usable_=false;
    if(!transfer(true,req.data(),req.size(),deadline,error)||!transfer(false,h,bound?52:28,deadline,error)) return false;
    if(std::memcmp(h,bound?"VFR3":"VFR2",4)!=0 || get32(h+4)!=2 || get64(h+8)!=0 || get32(h+16)!=0 || get32(h+20)!=2560 || get32(h+24)!=layer_) {error="invalid bridge shutdown"; return false;}
    if(bound) {
        const size_t id_size=identity_->run_id.size()+identity_->model_revision.size()+identity_->plan_id.size();
        if(get64(h+28)!=identity_->epoch || get16(h+36)!=identity_->run_id.size() ||
           get16(h+38)!=identity_->model_revision.size() || get16(h+40)!=identity_->plan_id.size() || get16(h+42)!=0 || get64(h+44)!=0) {
            error="bridge shutdown identity header mismatch"; return false;
        }
        std::vector<uint8_t> echoed(id_size);
        if(!transfer(false,echoed.data(),echoed.size(),deadline,error)) return false;
        if(!identity_matches(echoed.data(),*identity_)) { error="bridge shutdown identity mismatch"; return false; }
    }
    return true;
} catch (...) {
    usable_=false;
    error="bridge shutdown failure";
    return false;
}
}
