#include "rk_npu_autotune_cache.h"
#include "rk_npu_common.h"
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace rknpu2_matmul_open::tune {
std::string cache_path(uint64_t hash,const char* directory,const char* space) {
    std::string base;
    if(directory) base=directory;
    else {
        const char* xdg=std::getenv("XDG_CACHE_HOME");const char* home=std::getenv("HOME");
        if(xdg && *xdg) base=xdg;
        else if(home && *home) base=std::string(home)+"/.cache";
        else return {};
        base+='/';base+=space;
    }
    if(base.empty()) return {};
    std::ostringstream path;path.imbue(std::locale::classic());path<<base;
    if(base.back()!='/') path<<'/';
    path<<std::hex<<std::setfill('0')<<std::setw(16)<<hash<<".tune";
    return path.str();
}
int load_cache(const std::string& path,const std::function<bool(std::istream&)>& read) {
    std::ifstream in(path,std::ios::binary);
    if(!in) return RK_NPU_ERR_CACHE_MISS;
    in.imbue(std::locale::classic());
    if(!read(in)) return RK_NPU_ERR_CACHE_MISS;
    in>>std::ws;return in.eof()?RK_NPU_OK:RK_NPU_ERR_CACHE_MISS;
}
namespace {
bool parent_directories(const std::string& path) {
    const size_t slash=path.find_last_of('/');
    if(slash==std::string::npos) return true;
    const std::string parent=path.substr(0,slash);if(parent.empty()) return true;
    for(size_t pos=parent[0]=='/'?1:0;;) {
        pos=parent.find('/',pos);const auto part=parent.substr(0,pos);
        if(!part.empty() && ::mkdir(part.c_str(),0700)!=0 && errno!=EEXIST) return false;
        if(pos==std::string::npos) break; ++pos;
    }
    return true;
}
}
int save_cache(const std::string& path,const std::function<void(std::ostream&)>& write) {
    if(path.empty() || !parent_directories(path)) return RK_NPU_ERR_IO;
    static std::atomic<uint64_t> serial{0};
    const auto tmp=path+".tmp."+std::to_string((long long)::getpid())+"."+
                   std::to_string(serial.fetch_add(1,std::memory_order_relaxed));
    try {
        std::ofstream out(tmp,std::ios::binary|std::ios::trunc);
        if(!out) return RK_NPU_ERR_IO;
        out.imbue(std::locale::classic());write(out);out.close();
        if(!out || ::rename(tmp.c_str(),path.c_str())!=0) {::unlink(tmp.c_str());return RK_NPU_ERR_IO;}
    } catch(...) {::unlink(tmp.c_str());throw;}
    return RK_NPU_OK;
}
} // namespace rknpu2_matmul_open::tune
