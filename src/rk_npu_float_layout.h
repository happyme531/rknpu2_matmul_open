#pragma once
#include <cstdint>

namespace rknpu2_matmul_open::detail {
inline int float_panel_width(int layout) { return layout == 2 ? 8 : layout == 3 ? 16 : 0; }
inline uint64_t float_native_offset(int M, int channels, int row, int channel,
                                     int panel_rows, int atom = 8) {
    if (!panel_rows) return (uint64_t(channel / atom) * M + row) * atom + channel % atom;
    return uint64_t(row / panel_rows) * panel_rows * channels +
        (uint64_t(channel / atom) * panel_rows + row % panel_rows) * atom + channel % atom;
}
}
