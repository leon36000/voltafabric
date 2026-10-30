// Scalar reference for sampled SC117 layer-3 Q/K/V/output rows across a live trace.
// Adapted from check_sc117_qsa_r430.cpp. Shares GGML quantizers and decoders.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static bool unchanged(const struct stat &a, const struct stat &b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
static bool read_exact(int fd, void *p, size_t n, off_t off) {
    auto *b = static_cast<unsigned char *>(p);
    while (n) {
        const ssize_t got = pread(fd, b, n, off);
        if (got <= 0) return false;
        b += got; n -= static_cast<size_t>(got); off += got;
    }
    return true;
}
int main(int argc, char **argv) {
    if (argc != 3 && !(argc == 4 && std::strcmp(argv[3], "--negative") == 0)) return 1;
    const bool negative = argc == 4;
    const int shard = open(argv[1], O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    const int trace = open(argv[2], O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    struct stat s0{}, t0{}, s1{}, t1{};
    if (shard < 0 || trace < 0 || fstat(shard, &s0) || fstat(trace, &t0) ||
        !S_ISREG(s0.st_mode) || !S_ISREG(t0.st_mode) ||
        s0.st_size != 55063448064LL || t0.st_size != 92LL * 37380 * 4) return 2;
    ggml_cpu_init();
    double maxima[4]{}, maxratio[4]{}; int counts[4]{};
    std::vector<float> o(37380);
    for (int pos = 0; pos < 92; ++pos) {
        if (!read_exact(trace, o.data(), o.size()*4, off_t(pos)*37380*4)) return 3;
        if (negative && pos == 91) o[2564] += 0.5f;
        for (int p = 0; p < 4; ++p) {
            const auto type = p == 0 ? GGML_TYPE_IQ4_XS : (p == 3 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q6_K);
            const int cols = p == 3 ? 6144 : 2560, rows = p == 0 ? 12288 : (p == 3 ? 2560 : 512);
            const int base = p == 0 ? 2564 : (p == 1 ? 27140 : (p == 2 ? 28164 : 34820));
            const int64_t offset = p == 0 ? 4105151616LL : (p == 1 ? 4087363712LL : (p == 2 ? 4121864320LL : 4088439936LL));
            const auto *w = ggml_get_type_traits(type); const auto *c = ggml_get_type_traits_cpu(type);
            if (!w || !w->to_float || !c) return 4;
            const auto *aq = ggml_get_type_traits_cpu(c->vec_dot_type); const auto *ad = ggml_get_type_traits(c->vec_dot_type);
            if (!aq || !aq->from_float) return 4;
            std::vector<uint8_t> x(ggml_row_size(c->vec_dot_type, cols)), b(ggml_row_size(type, cols));
            std::vector<float> xd(cols), wd(cols);
            aq->from_float(o.data() + (p == 3 ? 28676 : 0), x.data(), cols);
            if (c->vec_dot_type == GGML_TYPE_Q8_K) {
                for (int n = 0; n < cols/256; ++n) {
                    float scale; std::memcpy(&scale, x.data()+n*292, 4);
                    for (int j=0; j<256; ++j) { int8_t v; std::memcpy(&v, x.data()+n*292+4+j, 1); xd[n*256+j]=scale*v; }
                }
            } else {
                if (!ad || !ad->to_float) return 5;
                ad->to_float(x.data(), xd.data(), cols);
            }
            const int ids[] = {0,1,127,255,511,1023,2047,4095,6143,8191,12287};
            for (int row : ids) {
                if (row >= rows) continue;
                if (!read_exact(shard, b.data(), b.size(), offset + off_t(row)*b.size())) return 6;
                w->to_float(b.data(), wd.data(), cols);
                double sum = 0; for (int j=0; j<cols; ++j) sum += double(wd[j])*xd[j];
                const double e=std::abs(double(o[base+row])-sum), tol=1e-5+1e-3*std::abs(sum);
                if (!std::isfinite(sum) || !std::isfinite(e) || e>tol) {
                    std::fprintf(stderr,"mismatch position=%d projection=%d row=%d error=%.9g tolerance=%.9g\n",pos,p,row,e,tol); return 7;
                }
                maxima[p]=std::max(maxima[p],e); maxratio[p]=std::max(maxratio[p],e/tol); ++counts[p];
            }
        }
    }
    if (fstat(shard, &s1) || fstat(trace, &t1) || !unchanged(s0,s1) || !unchanged(t0,t1)) return 8;
    close(shard); close(trace);
    for (int p=0;p<4;++p) std::printf("{\"projection\":%d,\"sampled\":%d,\"max_abs\":%.17g,\"max_tolerance_fraction\":%.17g}\n",p,counts[p],maxima[p],maxratio[p]);
    return 0;
}
