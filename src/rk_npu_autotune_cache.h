#pragma once
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>

namespace rknpu2_matmul_open::tune {
std::string cache_path(uint64_t hash,const char* directory,const char* default_namespace);
int load_cache(const std::string&,const std::function<bool(std::istream&)>&);
int save_cache(const std::string&,const std::function<void(std::ostream&)>&);
} // namespace rknpu2_matmul_open::tune
