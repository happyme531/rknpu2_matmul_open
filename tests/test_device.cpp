#include "../src/rk_npu_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <glob.h>
#include <sys/ioctl.h>
#include <vector>

namespace {

void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "device test failed: %s\n", message);
        std::exit(1);
    }
}

/* Native DRM_VERSION ABI, emulated independently of the RKNPU actions. */
struct DrmVersion {
    int major, minor, patchlevel;
    size_t name_len;
    char* name;
    size_t date_len;
    char* date;
    size_t desc_len;
    char* desc;
};

struct Device {
    const char* path;
    const char* driver = "rknpu";
    int open_error = 0;
    int drm_error = 0;
    uint32_t failed_action = 0;
    bool iommu = true;
    bool opened = false;
    int actions = 0;
    int closes = 0;
};

std::vector<Device> devices;
std::vector<char*> render_paths;
int glob_result = 0, glob_calls = 0, glob_frees = 0;
constexpr int fd_base = 1000;

void reset(std::initializer_list<Device> next) {
    for (const auto& device : devices)
        require(!device.opened, "device fd leaked");
    require(glob_calls == glob_frees, "discovery storage leaked");
    devices = next;
    render_paths.clear();
    glob_result = glob_calls = glob_frees = 0;
}

Device& device_for_fd(int fd) {
    require(fd >= fd_base && size_t(fd - fd_base) < devices.size(), "valid fd");
    auto& device = devices[size_t(fd - fd_base)];
    require(device.opened, "fd is still open");
    return device;
}

void expect_failure(const char* path, int error) {
    errno = 0;
    require(rk_npu_open(path) == nullptr, "open must fail");
    require(errno == error, "failure retains the cause in errno");
    for (const auto& device : devices)
        require(!device.opened, "failed open closes its fd");
}

} // namespace

extern "C" int __wrap_open(const char* path, int flags, ...) {
    require((flags & (O_RDWR | O_CLOEXEC)) == (O_RDWR | O_CLOEXEC),
            "device fd is read/write and close-on-exec");
    for (size_t i = 0; i < devices.size(); ++i) {
        auto& device = devices[i];
        if (std::strcmp(device.path, path) != 0) continue;
        if (device.open_error) { errno = device.open_error; return -1; }
        require(!device.opened, "one active fd per emulated device");
        device.opened = true;
        return fd_base + int(i);
    }
    errno = ENOENT;
    return -1;
}

extern "C" int __wrap_close(int fd) {
    auto& device = device_for_fd(fd);
    device.opened = false;
    ++device.closes;
    errno = EBADF; // Cleanup must not replace the original failure cause.
    return 0;
}

extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
    auto& device = device_for_fd(fd);
    va_list args;
    va_start(args, request);
    void* argument = va_arg(args, void*);
    va_end(args);
    if (request == _IOWR('d', 0x00, DrmVersion)) {
        if (device.drm_error) { errno = device.drm_error; return -1; }
        auto* version = static_cast<DrmVersion*>(argument);
        const size_t length = std::strlen(device.driver);
        std::memcpy(version->name, device.driver, std::min(version->name_len, length));
        version->name_len = length;
        return 0;
    }
    require(request == IOCTL_ACTION, "only read-only initialization ioctls");
    ++device.actions;
    auto* action = static_cast<rknpu_action*>(argument);
    if (action->flags == device.failed_action) { errno = EIO; return -1; }
    if (action->flags == RKNPU_GET_DRV_VERSION) action->value = 908;
    else {
        require(action->flags == RKNPU_GET_IOMMU_EN, "IOMMU query");
        action->value = device.iommu;
    }
    return 0;
}

extern "C" int __wrap_glob(const char* pattern, int, int (*)(const char*, int),
                             glob_t* nodes) {
    require(std::strcmp(pattern, "/dev/dri/renderD*") == 0, "render-only discovery");
    ++glob_calls;
    if (glob_result) return glob_result;
    for (const auto& device : devices)
        if (std::strncmp(device.path, "/dev/dri/renderD", 16) == 0)
            render_paths.push_back(const_cast<char*>(device.path));
    nodes->gl_pathc = render_paths.size();
    render_paths.push_back(nullptr);
    nodes->gl_pathv = render_paths.data();
    return nodes->gl_pathc ? 0 : GLOB_NOMATCH;
}

extern "C" void __wrap_globfree(glob_t* nodes) {
    ++glob_frees;
    *nodes = {};
}

int main() {
    reset({{"/dev/dri/renderD128", "panfrost"}, {"/dev/dri/renderD137"},
           {"/dev/dri/card1"}});
    auto* ctx = rk_npu_open(nullptr);
    require(ctx && ctx->fd == fd_base + 1, "find NPU at a non-default render number");
    require(ctx->driver_version == 908 && ctx->iommu_enabled, "queried metadata");
    require(devices[0].actions == 0 && devices[0].closes == 1,
            "skip GPU without sending RKNPU actions");
    require(!devices[2].opened && glob_calls == glob_frees, "release discovery list");
    rk_npu_close(ctx);

    reset({{"/dev/dri/card3"}});
    devices[0].iommu = false;
    ctx = rk_npu_open("/dev/dri/card3");
    require(ctx && !ctx->iommu_enabled && glob_calls == 0, "explicit primary node");
    rk_npu_close(ctx);

    reset({{"/dev/dri/renderD129"}});
    ctx = rk_npu_open("/dev/dri/renderD129");
    require(ctx && glob_calls == 0, "explicit render node");
    rk_npu_close(ctx);
    expect_failure("/dev/dri/missing", ENOENT);
    expect_failure("", ENOENT);
    require(glob_calls == 0, "explicit path never falls back to discovery");

    for (const char* driver : {"rockchip", "rknpu-other"}) {
        reset({{"/dev/dri/card0", driver}});
        expect_failure("/dev/dri/card0", ENODEV);
        require(devices[0].actions == 0, "reject other DRM drivers before actions");
    }
    reset({{"/dev/null"}});
    devices[0].drm_error = ENOTTY;
    expect_failure("/dev/null", ENOTTY);

    for (uint32_t action : {RKNPU_GET_DRV_VERSION, RKNPU_GET_IOMMU_EN}) {
        reset({{"/dev/dri/renderD129"}});
        devices[0].failed_action = action;
        expect_failure("/dev/dri/renderD129", EIO);
        expect_failure(nullptr, EIO);
    }
    reset({{"/dev/dri/renderD129"}, {"/dev/dri/renderD142"}});
    devices[0].failed_action = RKNPU_GET_DRV_VERSION;
    ctx = rk_npu_open(nullptr);
    require(ctx && ctx->fd == fd_base + 1 && devices[0].closes == 1,
            "discovery continues past a failed candidate");
    rk_npu_close(ctx);

    reset({{"/dev/dri/renderD128", "panfrost"}, {"/dev/dri/card1"}});
    expect_failure(nullptr, ENODEV);
    reset({});
    expect_failure(nullptr, ENODEV);
    reset({{"/dev/dri/renderD129"}, {"/dev/dri/renderD130", "panfrost"}});
    devices[0].open_error = EACCES;
    expect_failure(nullptr, EACCES);
    reset({});
    glob_result = GLOB_NOSPACE;
    expect_failure(nullptr, ENOMEM);
    reset({});
    std::puts("device discovery / identity / query failure cleanup PASS");
}
